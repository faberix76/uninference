# Disk-Backed KV Cache — Design Specification

## 1. Executive Summary

Add a `--disk-kv-cache` startup parameter to `llama.cpp` that redirects ALL KV cache storage from VRAM/RAM to disk. The KV cache behaves identically to the in-memory version: same API, same cell/slot semantics, same attention computation — but the underlying tensor data lives in files on a user-specified path instead of GPU or CPU buffers.

No tiering, no hot/warm/cold hierarchy. **Everything on disk, always.** The goal is to run inference at maximal context length (millions of tokens) on hardware where the full KV cache would not otherwise fit in available RAM+VRAM, at the cost of higher per-token latency (bounded by storage bandwidth).

---

## 2. Startup Parameter Specification

### 2.1. `struct llama_context_params` — New Fields

```c
struct llama_context_params {
    // ... existing fields ...

    // --disk-kv-cache
    // Path to a directory for storing KV cache files.
    // NULL or empty = in-memory KV cache (current behaviour).
    // When set, ALL KV cache data is stored in files under this directory.
    // The directory is created on init if it does not exist.
    const char * disk_cache_path;

    // --disk-kv-cache-max-size
    // Soft limit in bytes for the on-disk KV cache storage.
    // 0 = unlimited (default).
    // When exceeded, the garbage collector evicts the oldest cells.
    uint64_t disk_cache_max_size;

    // --disk-kv-cache-page-size
    // I/O granularity in bytes (default: 4096, must be power of 2).
    // Controls the alignment of reads/writes to disk.
    uint32_t disk_cache_page_size;

    // --disk-kv-cache-threads
    // Number of background I/O threads (default: 1, max: 4).
    // Used for async read-ahead and write-back.
    uint32_t disk_cache_threads;

    // --disk-kv-cache-prefetch
    // Number of cells to prefetch ahead of the current read position (default: 0).
    // Higher values trade RAM for lower latency on sequential access.
    uint32_t disk_cache_prefetch_cells;
};
```

### 2.2. CLI Example

```
./llama-cli \
    --model DeepSeek-R1-UD-IQ1_S-1.58bit-8k-turbo3.guf \
    --disk-kv-cache /mnt/fastssd/kv_cache/run_001 \
    --disk-kv-cache-max-size 500G \
    --disk-kv-cache-page-size 65536 \
    --disk-kv-cache-threads 2 \
    --disk-kv-cache-prefetch 256 \
    -n -1 --ctx-size 0  # max context from model
```

---

## 3. Architecture

### 3.1. New Abstraction Layer

```
llama_memory_i
  ├── llama_kv_cache                ← unchanged, in-RAM
  ├── llama_kv_cache_iswa           ← unchanged, in-RAM
  ├── llama_memory_recurrent        ← unchanged, in-RAM
  └── llama_kv_cache_disk           ← NEW: fully disk-backed
        ├── extends llama_memory_i
        ├── wraps a llama_kv_cache instance for metadata ONLY
        └── delegates tensor storage to DiskStore
              ├── DiskStore per backend buffer type (CPU/GPU)
              └── FileRegion: [file_path, offset, size]
```

`llama_kv_cache_disk` reuses ALL existing slot logic (`find_slot`, `apply_ubatch`, `seq_rm`, `seq_cp`, etc.) from `llama_kv_cache` — it inherits or wraps the metadata (`v_cells`, `v_heads`, `seq_to_stream`, etc.). The difference is:

- **No real backend buffers are allocated.** `ctxs_bufs` contains zero-size dummy buffers.
- Tensor `data` pointers are replaced with `FileRegion` descriptors.
- `ggml_backend_tensor_set` / `ggml_backend_tensor_get` calls are routed through the `DiskStore` I/O engine.

### 3.2. DiskStore Engine

```
DiskStore
  ├── FilePool
  │     ├── /path/to/cache/stream_0.bin   ← all layers, stream 0
  │     ├── /path/to/cache/stream_1.bin   ← all layers, stream 1
  │     └── ... (1 file per stream, or 1 file per stream per layer)
  ├── PageCache (optional, bounded LRU)
  │     └── [page_key → pinned_buffer]    ← hosts recently-read pages
  ├── IoUring / aio worker thread pool
  │     ├── submit(read/write) → future<size_t>
  │     └── completion callbacks
  ├── WriteQueue (coalescing)
  │     └── pending writes ordered by file offset
  └── MetadataJournal
        └── append-only log of cell allocation/free events
```

#### 3.2.1. File Layout (per stream)

One file per stream (or optionally one file per stream per layer for finer granularity):

```
stream_N.bin
┌──────────────────────────────────────────┐
│  FileHeader (4 KiB, zero-padded)         │
│   - magic: "TQDK"                        │
│   - version: 1                           │
│   - n_layer, n_stream, n_cells_max       │
│   - per-layer type, embd, stride info    │
│   - page_size                            │
├──────────────────────────────────────────┤
│  Cell Allocator Bitmap (1 bit per cell)   │
│   - 0 = free, 1 = allocated              │
│   - sized to n_cells_max bits,           │
│     rounded up to nearest page           │
├──────────────────────────────────────────┤
│  Cell Index Table                        │
│   - array of CellIndexEntry[n_cells_max] │
│   - each entry:                          │
│       - pos (llama_pos, -1 if free)      │
│       - seq_id bitset (32 bytes)         │
│       - offset_in_file (uint64_t)        │
│       - byte_count (uint32_t)            │
├──────────────────────────────────────────┤
│  Padding to page boundary                │
├──────────────────────────────────────────┤
│  Layer Data Blocks                       │
│   ┌────────────────────────────────────┐ │
│   │ BlockHeader per allocation          │ │
│   │  - layer_index                     │ │
│   │  - cell_count                      │ │
│   │  - cell_indices[]                  │ │
│   │  - data_size                       │ │
│   ├────────────────────────────────────┤ │
│   │ BlockData (row-major packed cells)  │ │
│   │  - K values: cell_0_row, cell_1... │ │
│   │  - V values: cell_0_row, cell_1... │ │
│   └────────────────────────────────────┘ │
│  ... more blocks ...                     │
└──────────────────────────────────────────┘
```

**Allocation strategy — Append-Only Log + Holes:**
- New cell writes are appended at the end of the file (sequential write = fast).
- When cells are freed (`seq_rm`, `clear`, eviction), their blocks become holes tracked in a free-list.
- The garbage collector consolidates holes when fragmentation exceeds a threshold.

#### 3.2.2. Cell ↔ File Offset Mapping

Each cell in the ring buffer has a `FileRegion`:

```cpp
struct FileRegion {
    int fd;                    // or file index
    uint64_t file_offset;      // byte offset in the stream file
    uint32_t byte_count;       // total bytes for K+V for this cell (all layers)
    uint32_t generation;       // incremented on re-write (detects stale cached pages)
};
```

The `FileRegion` is stored alongside the cell metadata in `v_cells` (or a parallel array).

---

## 4. Behaviour — Per Operation

### 4.1. Cache Initialisation

1. `disk_cache_path` is validated (created if missing, checked for write access).
2. Stream files are created (or reopened if resuming a previous session).
3. `FileHeader` is written with model/cache parameters.
4. `Cell Allocator Bitmap` is initialised to all-free.
5. `Cell Index Table` is initialised to all-pos=-1.

**No tensors are allocated.** The dummy `ggml_backend_buffer` contains zero bytes.

### 4.2. `find_slot()`

Identical to the in-memory version — operates purely on `v_cells` metadata (positions, seq_ids). Returns `slot_info` with stream and cell indices. No disk I/O occurs during slot finding.

### 4.3. `apply_ubatch()` — Writing New Cells

For each cell in the slot:

1. **Serialize K tensor slice** for the cell (all layers) into a contiguous buffer in a scratch memory pool.
2. **Serialize V tensor slice** (same).
3. **Allocate file offset:** append at current end-of-file (or reuse a hole of sufficient size from the free-list).
4. **Write BlockHeader** (layer_index, cell_count, cell_index, data_size).
5. **Write K+ V data** via `IoUring` / `pwritev` (submitted to write queue).
6. **Update Cell Index Table** in the file header area (or defer to batch commit).
7. **Update in-memory `FileRegion`** for the cell.

**Coalescing:** If multiple cells in the same ubatch target the same stream, their data is coalesced into a single sequential write (fewer syscalls, full bandwidth utilisation).

### 4.4. `get_k()` / `get_v()` — Reading for Attention

These methods return `ggml_tensor *` views that point to **file-backed memory** (mmap) or **dynamically populated scratch buffers**.

**Option A — mmap (recommended for read-mostly workloads):**
- The stream file is `mmap`'d with `MAP_SHARED | MAP_POPULATE` for the used region.
- `ggml_tensor->data` points directly into the mmap'd range.
- The OS handles page-in/page-out transparently. The `--disk-kv-cache-page-size` parameter sets `mmap` alignment.
- **Caveat:** On 32-bit systems or with extremely large contexts (>100B), the virtual address space may be exhausted. In that case, fall back to Option B.

**Option B — explicit pread (bounded page cache):**
- Before entering the attention computation (`build_graph_shift` / `apply_ubatch`), issue a batch of `preadv` calls for all cells referenced by the current `slot_info`.
- Data lands in a pinned page-cache buffer (LRU, max `--disk-kv-cache-page-cache-mb` MiB).
- `ggml_tensor->data` points into the page-cache slot.
- After attention completes, the page-cache slot is released back to the LRU pool.

**Prefetch (`--disk-kv-cache-prefetch`):**
- After reading cell N (autoregressive decode), issue an async `preadv` for cells [N+1, N+prefetch_cells] in the background.
- By the time the next decode step runs, data is (hopefully) in the page cache.

### 4.5. `update()` — K-Shift

RoPE-based K-shift modifies K values in-place. On disk:

1. **Read** the affected cells from disk into a scratch buffer.
2. **Apply** the RoPE shift in CPU (or GPU, if a small compute buffer exists).
3. **Mark** the cells as `dirty` in the `WriteQueue`.
4. The write-back is flushed either synchronously (before the next `get_k()`) or asynchronously (background thread, similar to `fsync`-friendly write-back).

Because K-shift is rare (only during cache compaction after sequence removal), the overhead is acceptable.

### 4.6. `seq_rm()` / `seq_keep()` — Sequence Management

These operate on metadata only (removing a seq_id from a cell's bitset). If the cell becomes empty (`seq.none()`):

1. The cell's `FileRegion` is enqueued in a **free-list** (hole in the file).
2. The `Cell Allocator Bitmap` bit is cleared.
3. The `Cell Index Table` entry's `pos` is set to -1.

**The file space is NOT immediately reclaimed** (no `fallocate(FALLOC_FL_PUNCH_HOLE)` — that would fragment the file and hurt append performance). Reclamation happens during garbage collection (Section 5).

### 4.7. `clear(true)` — Reset

1. Truncate all stream files to the header size (`ftruncate`).
2. Zero the `Cell Allocator Bitmap`.
3. Zero the `Cell Index Table`.
4. Reset in-memory `v_cells` and `v_heads`.
5. Flush and discard the `WriteQueue`.
6. Invalidate the page cache.

### 4.8. `state_write()` / `state_read()`

Reuses the existing `llama_io_write_i` / `llama_io_read_i` abstraction.

- `state_write()` serialises cell metadata only (the on-disk data is already on disk). The resulting buffer is tiny (just positions and seq IDs).
- `state_read()` restores metadata. The cell data stays where it is on disk.
- For **copy between contexts** (e.g., `state_seq_get_data` / `state_seq_set_data`): the cell data is read from the source file and written to the destination file. This is a file-to-file copy, never going through VRAM.

---

## 5. Garbage Collection & Defragmentation

### 5.1. Triggers

- **Free space threshold:** When the number of free cells drops below `max_cells * 0.1` (configurable).
- **Fragmentation ratio:** When the ratio of allocated-file-size to logical-used-size exceeds 2.0 (i.e., >50% of the file is holes).
- **Explicit:** `llama_kv_cache_disk_defrag()` API call.
- **Periodic:** Timer-based (every N tokens or every M seconds, configurable).

### 5.2. Algorithm — Mark-Compact

```
Phase 1: MARK
  ─ Scan cell index table, mark all allocated cells.
  ─ Build a new sequential packing: [cell_0, cell_1, ..., cell_N]
    sorted by position (or seq_id, configurable).

Phase 2: COMPACT
  ─ Create a temporary file "stream_N.compact".
  ─ Write new Cell Allocator Bitmap + Cell Index Table.
  ─ Iterate allocated cells in packing order:
       read cell data from old offset,
       write to new file at sequential offset.
  ─ Record old-offset → new-offset mapping.

Phase 3: SWAP
  ─ fsync compact file.
  ─ rename("stream_N.compact", "stream_N.bin").
  ─ Update all in-memory FileRegion offsets via the mapping table.

Phase 4: RECLAIM
  ─ For layers with multiple streams, repeat for each stream.
  ─ Remove old file handles.
  ─ Reset in-memory hole free-list.
```

**Duration:** For a 100 GiB KV cache file on a modern NVMe (~5 GiB/s read, ~3 GiB/s write), compaction takes ~60 seconds. During compaction, reads still work (old file is still accessible until rename).

### 5.3. Concurrent Access During GC

All I/O goes through the `DiskStore` which has a read-write lock:
- Multiple readers can access the current file concurrently.
- GC needs an exclusive write lock.
- If a decode step is in progress, GC waits for it to complete.
- GC yields the lock periodically (every 64 MiB processed) to avoid starving decode.

---

## 6. Write-Ahead Log (WAL) for Crash Safety

To prevent corruption on crash, all mutations go through a WAL:

1. **Write to WAL** (`stream_N.wal`): append-only log of `(cell_index, old_offset_old_gen, new_data)`.
2. **Write to main file** (async, via write queue).
3. **Trim WAL:** once main file writes are `fsync`'d, truncate the WAL to the last checkpoint.

On next init, if a WAL exists:
- Replay pending writes.
- Or detect corruption and fall back to last-good checkpoint.

---

## 7. Integration with Existing Features

### 7.1. TurboQuant

The rotation matrices (`turbo_rotation`, `turbo_rotation_inv`, `turbo_innerq_scale_inv`) are small (128×128 F32 = 64 KiB each). They are:
- Stored in the file header region (not per-cell).
- Loaded into a pinned CPU buffer on init.
- If `offload_kqv=true`, they are copied to GPU on each decode step (or kept in a tiny GPU allocation).

The InnerQ calibration (`turbo_innerq_needs_tensor_update`) must write through to the file header when updated.

### 7.2. Layer-Adaptive KV Cache

Each layer may have a different quantization type (`TURBO_LAYER_ADAPTIVE`). The `FileHeader` contains a per-layer type descriptor. The serialized block size varies per layer. `DiskStore` reads/writes one layer at a time, using the correct `ggml_row_size()` for each.

### 7.3. Unified vs Non-Unified Cache

- **Unified** (`kv_unified=true`, `n_stream=1`): Single stream file. All sequences share the same file.
- **Non-Unified** (`n_stream=n_seq_max`): One file per stream. `seq_to_stream` maps sequence → stream file.

### 7.4. SWA / ISWA / Hybrid

`llama_kv_cache_iswa` wraps two `llama_kv_cache` instances (base + SWA). In disk mode:
- Both caches become `llama_kv_cache_disk` instances.
- They share the same `DiskStore` but use independent stream files (or the same file with different name prefixes).

`llama_memory_hybrid` (attention + recurrent) works the same way — recurrent states are small and stay in RAM (they don't scale with context length).

### 7.5. `no_alloc` Mode

When `--disk-kv-cache` is active, `no_alloc` is automatically set to `true`. No GPU/CPU buffer is allocated. The dummy buffers satisfy the scheduler's expectation of having a `buffer` pointer.

---

## 8. Performance Characteristics (Upper Bounds)

| Metric | Value | Notes |
|--------|-------|-------|
| Read bandwidth (single NVMe) | ~7 GiB/s | Sequential, `preadv` with 4 threads |
| Write bandwidth (single NVMe) | ~5 GiB/s | Append-only; coalesced writes |
| Write amplification (GC) | 2× | Best case; 3-4× under pathological fragmentation |
| Read latency (first byte) | ~5 µs | NVMe + io_uring; page cache miss |
| Read latency (page cache hit) | ~0.1 µs | Kernel page cache |
| Tokens/sec (3B model, 4K ctx) | ~5 tok/s | vs ~80 tok/s in-memory (CPU) |
| Tokens/sec (3B model, 128K ctx) | ~3 tok/s | vs OOM in-memory |
| Tokens/sec (70B model, 8K ctx) | ~0.5 tok/s | vs ~5 tok/s in-memory; enables large contexts that would otherwise OOM |
| page-cache RAM | 0–few GB | Configurable via `--disk-kv-cache-page-cache-mb` |

The primary use case is **enabling inference at context lengths that exceed physical RAM+VRAM**, not matching in-memory speeds.

---

## 9. Files to Create / Modify (for Implementation)

| Action | File | Reason |
|--------|------|--------|
| **NEW** | `src/llama-disk-store.h` | DiskStore engine: FilePool, PageCache, IoUring, WriteQueue |
| **NEW** | `src/llama-disk-store.cpp` | Implementation |
| **NEW** | `src/llama-kv-cache-disk.h` | `llama_kv_cache_disk`: wraps `llama_kv_cache` metadata + DiskStore |
| **NEW** | `src/llama-kv-cache-disk.cpp` | Implementation |
| **NEW** | `src/llama-disk-gc.h/.cpp` | Garbage collector: mark-compact, defrag, hole management |
| **NEW** | `src/llama-disk-wal.h/.cpp` | Write-ahead log for crash safety |
| MODIFY | `include/llama.h` | Add `disk_cache_*` fields to `llama_context_params` |
| MODIFY | `src/llama-context.cpp` | Create `llama_kv_cache_disk` when `disk_cache_path` is set |
| MODIFY | `src/llama-kv-cache.cpp` | Add `FileRegion` field to cell tracking (optional, conditional on `ifdef` or flag) |
| MODIFY | `common/common.cpp` | Parse `--disk-kv-cache-*` CLI flags → `llama_context_params` |
| MODIFY | `src/CMakeLists.txt` | Add new source files |
| MODIFY | `src/llama-kv-cache-iswa.h/.cpp` | Support disk-backed inner caches |
| MODIFY | `src/llama-memory-hybrid.cpp` | Support disk-backed KV portion |

---

## 10. Limitations & Future Directions

### 10.1. Known Limitations

- **Multi-GPU:** The disk cache does not exploit GPU direct storage (GDS). All I/O goes through CPU RAM.
- **Context shift (K-shift) on large ranges:** Shifting 1M cells requires reading, rotating, and rewriting 1M cells. This could take seconds. Mitigation: batch K-shift in blocks and pipeline with decode.
- **No encryption:** Data on disk is in plaintext. Future: add `--disk-kv-cache-encrypt` with AES-GCM.
- **Memory-mapped file limit:** On Linux, `vm.max_map_count` may need to be increased for extremely large mmap regions.
- **WAL growth:** Under heavy write load, the WAL may grow unboundedly. The GC should checkpoint aggressively.

### 10.2. Future Extensions

- **Hybrid mode:** `--disk-kv-cache-tiered` — keep last N tokens in RAM/VRAM, oldest on disk.
- **Distributed KV cache:** `--disk-kv-cache-distributed` — shard cells across multiple machines via RDMA/NVMe-oF.
- **Compressed page cache:** Use transparent zstd compression for the in-memory page cache (trade CPU for RAM).
- **Persistence across restarts:** `--disk-kv-cache-session SESSION_ID` — resume a previous session by reusing the same stream files.
- **Direct Storage (GDS/CXL):** Bypass CPU RAM entirely, read/write GPU ↔ NVMe.
