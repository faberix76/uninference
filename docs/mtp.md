# MTP (Multi-Token Prediction) — Porting Plan

## Obiettivo

Portare la PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) (`llama + spec: MTP Support`, am17an)
nella distribuzione turboquant modificando `update-llama.sh` in modo che:
1. Cloni `llama-cpp-turboquant` (come già fa)
2. Applichi le modifiche MTP sopra la base appena clonata
3. Il modello **non-MTP** continui a funzionare senza cambi di parametri
4. Il modello **MTP** (`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`) funzioni con `--spec-type mtp`
5. I parametri di lancio siano compatibili con entrambe le varianti

---

## 1. Analisi della PR #22673

### Statistiche
- 18 commit, +1844/−348 righe, 49 file
- Ancora **aperta** (non mergiata in master di ggml-org/llama.cpp)
- Branch: `am17an:mtp-clean`, base: `ggml-org:master`

### Dipendenze
- Dipende dalla PR [#22400](https://github.com/ggml-org/llama.cpp/pull/22400) (`llama: allow partial seq_rm for GDN models for speculative decoding`)
  — **già inclusa in turboquant** (il codice ha `n_rs` e rollback parziale).

### Cosa la PR aggiunge (rispetto a master ggml-org)
1. **Nuovo tipo speculative**: `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`
2. **Nuova architettura**: `qwen35_mtp` (MTP head = 1 transformer layer + proiezioni)
3. **Nuovo graph type**: `LLM_GRAPH_TYPE_DECODER_MTP`
4. **Pre-norm embeddings hook**: permette al MTP di consumare le hidden states dopo ogni ubatch
5. **`n_rs_seq`**: bounded partial sequence removal per GDN rollback con MTP
6. **GGDL state snapshots**: `ggml_gated_delta_net` con supporto K>1 (per-token snapshot)
7. **Server**: `--spec-type mtp`, auto-caricamento MTP head dallo stesso GGUF
8. **Download**: auto-scoperta file `mtp-*.gguf` su HuggingFace
9. **GGUF conversion**: `convert_hf_to_gguf.py --mtp` per generare GGUFs MTP

---

## 2. Cosa turboquant HA GIA' (e NON va toccato)

| Componente | File | Stato |
|------------|------|-------|
| `nextn_predict_layers` KV/hparams | `llama-hparams.cpp/h`, `llama-model.cpp` | ✅ Già presente |
| Tensori NEXTN in arch | `llama-arch.cpp/h` | ✅ Già presente (ma come `LAYER_OUTPUT`) |
| Qwen3.5 model (no MoE) | `src/models/qwen35.cpp` | ✅ Già presente |
| Qwen3.5 MoE model | `src/models/qwen35moe.cpp` | ✅ Già presente |
| Delta-net-base | `src/models/delta-net-base.cpp` | ✅ Già presente |
| Qwen3 Next | `src/models/qwen3next.cpp` | ✅ Già presente |
| GDN CUDA kernel | `ggml/src/ggml-cuda/gated_delta_net.cu` | ✅ Già presente |
| `n_rs` recurrent state rollback | `llama-graph.cpp` | ✅ Già presente |
| `llama-memory-recurrent.*` | `src/llama-memory-recurrent.*` | ✅ Già presente |
| `llama-memory-hybrid*` | `src/llama-memory-hybrid*` | ✅ Già presente |
| GGUF constants per NEXTN | `gguf-py/gguf/constants.py` | ✅ Già presente |
| `common/speculative.cpp/h` | `common/speculative.cpp/h` | ✅ Già presente |
| `llama-ext.h` staging API | `src/llama-ext.h` | ✅ Già presente |
| `done_getting_tensors()` | `src/llama-model-loader.*` | ✅ Già presente |

---

## 3. Cosa MANCA (da portare)

### 3.1 Modifiche a file esistenti

Ogni modifica è elencata con il file, il cambiamento specifico e il rischio di regressione.

#### A. `common/common.h`
- **Cambiamento**: Aggiungere `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` all'enum (dopo `COMMON_SPECULATIVE_TYPE_EAGLE3`), aggiungere `uint32_t need_n_rs_seq() const` in `common_params_speculative`, aggiungere `COMMON_CONTEXT_SEQ_RM_TYPE_RS = 3`
- **Rischio**: BASSO (solo nuovi valori enum, nulla viene rimosso)
- **Attenzione**: Turboquant usa `COMMON_SPECULATIVE_TYPE_DRAFT` e `COMMON_SPECULATIVE_TYPE_EAGLE3` — la PR usa `DRAFT_SIMPLE` e `DRAFT_EAGLE3`. Il nuovo valore `DRAFT_MTP` si aggiunge semplicemente dopo. Nessun rename necessario.

#### B. `common/speculative.h`
- **Cambiamento**: Aggiungere `bool common_speculative_need_embd(common_speculative * spec)` dichiarazione
- **Rischio**: BASSO (solo nuova dichiarazione)

#### C. `common/speculative.cpp`
- **Cambiamento**: Aggiungere `"draft-mtp"` alla mappa di parsing, aggiungere l'intera classe `common_speculative_state_draft_mtp` (~300 righe), aggiungere `common_speculative_need_embd()` implementazione
- **Rischio**: MEDIO (nuova funzionalità, non tocca codice esistente)
- **Dettaglio**: La classe mantiene stato per-seq di hidden states (`pending_h`, `verify_h`, `last_n_drafted`) e interagisce con il contesto MTP via `llama_set_embeddings_pre_norm()` / `llama_get_embeddings_pre_norm_ith()`

#### D. `include/llama.h`
- **Cambiamento**: Aggiungere `enum llama_context_type { LLAMA_CONTEXT_TYPE_DEFAULT = 0, LLAMA_CONTEXT_TYPE_MTP = 1 }`, aggiungere `uint32_t n_rs_seq` e `enum llama_context_type ctx_type` a `llama_context_params`, aggiungere `LLAMA_API uint32_t llama_n_rs_seq(const struct llama_context * ctx)`
- **Rischio**: BASSO (solo nuove API, retrocompatibile — nuovi campi inizializzati a 0/default)

#### E. `src/llama-ext.h`
- **Cambiamento**: Aggiungere staging API: `llama_set_embeddings_pre_norm()`, `llama_get_embeddings_pre_norm()`, `llama_get_embeddings_pre_norm_ith()`
- **Rischio**: BASSO (staging API, nuovo codice)

#### F. `src/llama-cparams.h`
- **Cambiamento**: Aggiungere campi `uint32_t n_rs_seq`, `bool embeddings_pre_norm`, `enum llama_context_type ctx_type`
- **Rischio**: BASSO (solo nuovi campi su struct)

#### G. `src/llama-hparams.h`
- **Cambiamento**: Aggiungere `bool kv_only_nextn = false` a `llm_hparams`
- **Rischio**: BASSO

#### H. `src/llama-hparams.cpp`
- **Cambiamento**: Modificare `has_kv()` per rispettare `kv_only_nextn` — solo gli ultimi `nextn_predict_layers` layer hanno KV cache
- **Rischio**: BASSO (flag default false, comportamento inalterato)

#### I. `src/llama-arch.cpp`
- **Cambiamento**: Cambiare la classificazione dei tensori `NEXTN_*` da `LLM_TENSOR_LAYER_OUTPUT` a `LLM_TENSOR_LAYER_REPEATING` (così vengono indicizzati come `blk.%d.nextn.*` con `bid` = layer number). Aggiungere `bool llm_arch_supports_rs_rollback()` per QWEN35/QWEN35MOE.
- **Rischio**: MEDIO — il cambio da `OUTPUT` a `REPEATING` cambia come i tensori vengono salvati/caricati nel GGUF. I GGUFs MTP hanno `blk.40.nextn.*`, e con `OUTPUT` venivano ignorati (commento: "NextN/MTP tensors are currently ignored"). Con `REPEATING` vengono caricati correttamente.
- **Verifica regressione**: I GGUFs non-MTP non hanno tensori NEXTN, quindi il caricamento è inalterato.

#### J. `src/llama-arch.h`
- **Cambiamento**: Aggiungere dichiarazione `bool llm_arch_supports_rs_rollback()`
- **Rischio**: BASSO

#### K. `src/llama-context.h`
- **Cambiamento**: Aggiungere metodi `get_embeddings_pre_norm()`, `get_embeddings_pre_norm_ith()`, `set_embeddings_pre_norm(bool)`, membro `buffer_view<float> embd_pre_norm`
- **Rischio**: BASSO

#### L. `src/llama-context.cpp` (modifica SIGNIFICATIVA)
- **Cambiamento**:
  1. `ctx_type_to_graph_type()` — mappa `LLAMA_CONTEXT_TYPE_DEFAULT → LLM_GRAPH_TYPE_DEFAULT`, `LLAMA_CONTEXT_TYPE_MTP → LLM_GRAPH_TYPE_DECODER_MTP`
  2. Costruttore: consuma `params.n_rs_seq`, valida con `llm_arch_supports_rs_rollback()`, setta `cparams.embeddings_pre_norm`
  3. Nuovi metodi: `get_embeddings_pre_norm()`, `get_embeddings_pre_norm_ith()`, `set_embeddings_pre_norm()`
  4. `encode()`: rilassa GGML_ASSERT per accettare batch con embd (MTP), estrae `t_h_pre_norm` dal graph result
  5. `decode()`: usa `ctx_type_to_graph_type()` invece di hardcoded `LLM_GRAPH_TYPE_DECODER`
  6. `output_reserve()`: alloca `embd_pre_norm` buffer
  7. `output_reorder()`: riordina `embd_pre_norm` insieme a logits/embd
  8. `llama_context_default_params()`: inizializza `n_rs_seq = 0`, `ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT`
  9. C API: `llama_n_rs_seq()`, `llama_set_embeddings_pre_norm()`, `llama_get_embeddings_pre_norm_ith()`
- **Rischio**: ALTO — modifiche al core di `encode()`/`decode()` possono impattare tutti i modelli. Le nuove funzionalità sono conditionali su `ctx_type != DEFAULT`.
- **Mitigazione**: Tutte le modifiche sono conditionali su `cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP` o su `cparams.embeddings_pre_norm == true`. Il default è `DEFAULT`/`false`.

#### M. `src/llama-graph.h`
- **Cambiamento**: Aggiungere `LLM_GRAPH_TYPE_DECODER_MTP` a `enum llm_graph_type`, aggiungere `ggml_tensor * t_h_pre_norm = nullptr` e getter a `llm_graph_result`
- **Rischio**: BASSO

#### N. `src/llama-graph.cpp`
- **Cambiamento**: In `build_rs()`, usare `s->ne[1]` invece di parametro `rs_size` per supportare stati wide (con rollout)
- **Rischio**: MEDIO — cambiamento a `build_rs()` chiamato da tutti i ricorrenti. Verificare che `s->ne[1]` sia sempre corretto.

#### O. `src/llama-memory.h`
- **Cambiamento**: Aggiungere `enum llama_context_type ctx_type` a `llama_memory_params`
- **Rischio**: BASSO

#### P. `src/llama-memory-recurrent.cpp/h`
- **Cambiamento**: Aggiungere supporto `n_rs_seq` per bounded partial seq_rm. Aggiungere `set_rs_idx()` per per-token state snapshot.
- **Rischio**: MEDIO — modifiche al core ricorrente. Con `n_rs_seq = 0` comportamento invariato.

#### Q. `src/llama-memory-hybrid.cpp/h` e `iswa` variant
- **Cambiamento**: Passare `n_rs_seq` e `ctx_type` ai costruttori interni
- **Rischio**: BASSO (solo propagazione parametri)

#### R. `src/llama-model.cpp`
- **Cambiamento**: In `llama_init_from_model()`: validare che MTP context type abbia `nextn_predict_layers > 0`. Nella creazione memoria: se `ctx_type == MTP`, creare `llama_memory_recurrent` pura (no hybrid wrapper) per il contesto MTP. Gestire `done_getting_tensors(partial)` per caricamento parziale.
- **Rischio**: MEDIO — conditional su ctx_type

#### S. `src/llama-model-loader.cpp/h`
- **Cambiamento**: Aggiungere `done_getting_tensors(bool partial)` per caricamento parziale dei trunk tensors (utile per GGUF MTP standalone)
- **Rischio**: BASSO

#### T. `src/models/qwen35.cpp` (modifica SIGNIFICATIVA)
- **Cambiamento**:
  1. `load_arch_hparams()`: leggere `nextn_predict_layers`, escludere MTP layer dal conteggio layer transformer principali, marcare layer ricorrenti correttamente
  2. `load_arch_tensors()`: refactor in `load_block_trunk()` e `load_block_mtp()` — MTP layer carica full attention + FFN + tensori NextN
  3. `mtp_only` mode: se `nextn_predict_layers > 0` e trunk tensors assenti (GGUF standalone MTP), trunk diventa `TENSOR_NOT_REQUIRED`
  4. `graph::graph()`: loop solo su `n_transformer_layers`, aggiungere `cb(cur, "h_pre_norm", -1)` e `res->t_h_pre_norm = cur`
  5. Nuovo `graph_mtp` constructor (~160 righe): `LLM_GRAPH_TYPE_DECODER_MTP` — esegue solo il blocco MTP
- **Rischio**: ALTO — modifiche al caricamento tensori e al grafo. `nextn_predict_layers == 0` = comportamento invariato.

#### U. `src/models/qwen35moe.cpp`
- **Cambiamento**: Stessi pattern di `qwen35.cpp` ma con FFN MoE nel `graph_mtp`
- **Rischio**: ALTO

#### V. `src/models/delta-net-base.cpp`
- **Cambiamento**: Aggiungere `keep_rs()` (determina quando salvare snapshot stato), refactor `build_layer_attn_linear()` in `build_conv_state()` + `build_recurrent_attn()` metodi base. `build_recurrent_attn()` esegue `ggml_gated_delta_net` con `K = n_rs_seq + 1`.
- **Rischio**: ALTO — refactor di `build_layer_attn_linear()` chiamata da tutti i modelli DeltaNet (Qwen3.5, Qwen3.6, ecc.)
- **Mitigazione**: `keep_rs()` restituisce `false` con `n_rs_seq == 0` (default), comportamento invariato.

#### W. `src/models/models.h`
- **Cambiamento**: Dichiarare `struct graph_mtp` per `llama_model_qwen35` e `llama_model_qwen35moe`
- **Rischio**: BASSO

#### X. `src/models/qwen3next.cpp`
- **Cambiamento**: Refactor per usare `build_conv_state()` / `build_recurrent_attn()` da `delta-net-base`
- **Rischio**: ALTO — stesse considerazioni del punto V

#### Y. `common/common.cpp`
- **Cambiamento**: In `common_context_can_seq_rm()`: aggiungere check `llama_n_rs_seq(ctx) > 0` e retournare `COMMON_CONTEXT_SEQ_RM_TYPE_RS`. In `common_context_params_to_llama()`: settare `cparams.n_rs_seq = params.speculative.need_n_rs_seq()`
- **Rischio**: BASSO

#### Z. `common/arg.cpp`
- **Cambiamento**: Aggiungere `found_mtp`/`mtp` a `handle_model_result`, propagare `download_result.mtp_path`, auto-detect MTP spec type per auto-set draft model path
- **Rischio**: BASSO

#### AA. `common/download.cpp` / `download.h`
- **Cambiamento**: Refactor `find_best_mmproj()` in `find_best_sibling(keyword)`, aggiungere `find_best_mtp()` (keyword `"mtp-"`), aggiungere campi MTP a `hf_plan` / `download_model_result`
- **Rischio**: MEDIO — refactor di `find_best_mmproj()`; testare che mmproj discovery continui a funzionare

#### BB. `ggml/` backend changes
- **Cambiamento**: GDN state snapshots per K>1 in:
  - `ggml/src/ggml.c` — validazione stato 3D `(S_v*S_v*H, K, n_seqs)`
  - `ggml/src/ggml-cpu/ggml-cpu.c` — scratch size per K>1
  - `ggml/src/ggml-cpu/ops.cpp` — per-token snapshot scrive su slot di output
  - `ggml/src/ggml-cuda/gated_delta_net.cu` — template parameter `keep_rs_t`
  - `ggml/src/ggml-metal/` — analogo
  - `ggml/src/ggml-vulkan/` — analogo
- **Rischio**: ALTO — modifiche ai backend di computazione. K==1 (default) comportamento invariato.
- **Mitigazione**: Tutti i cambiamenti sono conditionali su `K > 1`. Con `K == 1` il comportamento è identico.

#### CC. `convert_hf_to_gguf.py`
- **Cambiamento**: Aggiungere `_Qwen35MtpMixin` con logica di filtering tensori (+`mtp_num_hidden_layers` a block_count), parametri GGUF (`nextn_predict_layers`), naming `mtp-` prefix, remap tensori `mtp.* → blk.*.nextn.*`
- **Rischio**: BASSO (solo script di conversione)

#### DD. `gguf-py/gguf/constants.py`
- **Cambiamento**: Aggiungere tensori `NEXTN_*` alle liste `MODEL_ARCH.QWEN35` e `MODEL_ARCH.QWEN35MOE`
- **Rischio**: BASSO (nuovi tensori opzionali — `TENSOR_NOT_REQUIRED`)
- **Verifica**: turboquant ha già i tensori NEXTN in queste liste? **SÌ** (righe 2884-2889 e 2915-2920). Nessun cambiamento necessario!

#### EE. `tools/server/server-context.cpp`
- **Cambiamento**: Caricamento MTP head da GGUF con `override_arch = "qwen35_mtp"`, gestione `need_embd()` nel speculative loop, init MTP speculative type, gestione `COMMON_CONTEXT_SEQ_RM_TYPE_RS`
- **Rischio**: MEDIO — modifiche al loop di generazione server, ma conditionali su spec type

---

## 4. Modifica a `update-llama.sh`

### Strategia

Invece di applicare le modifiche direttamente su `uninference/` (che viene rimpiazzato ad ogni update),
si crea un **patchset** (serie di patch `.patch` o un singolo script di apply) nella directory
`patches/mtp/` dentro turboquant. Dopo aver clonato e copiato i file in `uninference/`,
lo script applica le patch.

### Struttura nuove directory

```
turboquant/
├── docs/mtp.md                ← questo file
├── patches/
│   └── mtp/
│       ├── apply-mtp.sh       ← script che applica TUTTE le patch in ordine
│       ├── 001-common-common-h.patch
│       ├── 002-common-speculative.patch
│       ├── 003-llama-h.patch
│       ├── 004-llama-ext-h.patch
│       ├── 005-llama-cparams-h.patch
│       ├── 006-llama-hparams.patch
│       ├── 007-llama-arch.patch
│       ├── 008-llama-context.patch
│       ├── 009-llama-graph.patch
│       ├── 010-llama-memory.patch
│       ├── 011-llama-model.patch
│       ├── 012-llama-model-loader.patch
│       ├── 013-models-qwen35.patch
│       ├── 014-models-qwen35moe.patch
│       ├── 015-models-delta-net-base.patch
│       ├── 016-models-models-h.patch
│       ├── 017-models-qwen3next.patch
│       ├── 018-common-common-cpp.patch
│       ├── 019-common-arg.patch
│       ├── 020-common-download.patch
│       ├── 021-ggml-backend.patch        ← CPU + CUDA + Metal + Vulkan
│       ├── 022-convert-hf-to-gguf.patch
│       └── 023-server-context.patch
```

### Modifica a `update-llama.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Cloning llama-cpp-turboquant ==="
git clone https://github.com/TheTom/llama-cpp-turboquant "$ROOT/llama-cpp-turboquant"

echo "=== Removing .git ==="
rm -rf "$ROOT/llama-cpp-turboquant/.git"

echo "=== Emptying uninference/ ==="
rm -rf "$ROOT/uninference/" && mkdir "$ROOT/uninference/"

echo "=== Moving contents to uninference/ ==="
rsync -a "$ROOT/llama-cpp-turboquant/" "$ROOT/uninference/"

echo "=== Removing source folder ==="
rm -rf "$ROOT/llama-cpp-turboquant"

echo "=== Removing unwanted files from uninference/ ==="
(cd "$ROOT/uninference" && rm -rf \
    .pre-commit-config.yaml \
    .gemini .github .docs .gitignore .git .gitattributes .gitmodules \
    AGENTS.md CLAUDE.md AUTHORS bench* CONTRIBUTING.md LICENSE \
    README.md SECURITY.md)

echo "=== Copying docker-files/ ==="
rsync -a "$ROOT/docker-files/" "$ROOT/uninference/"

# ─── NUOVO: Applica patch MTP ─────────────────────────────────
echo "=== Applying MTP patches (PR #22673) ==="
MTP_PATCHES="$ROOT/patches/mtp"
if [ -d "$MTP_PATCHES" ]; then
    # apply-mtp.sh si aspetta di girare dentro uninference/
    (cd "$ROOT/uninference" && bash "$MTP_PATCHES/apply-mtp.sh")
    echo "=== MTP patches applied successfully ==="
else
    echo "=== WARNING: patches/mtp/ not found, skipping MTP patches ==="
fi
# ─── FINE NUOVO ───────────────────────────────────────────────

echo "=== Done ==="
```

### Script `apply-mtp.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
# apply-mtp.sh — Applica tutte le patch MTP in ordine topologico
# Deve essere eseguito dalla root di uninference/

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Applying MTP patches in order..."
for patch in \
    001-common-common-h.patch \
    002-common-speculative.patch \
    003-llama-h.patch \
    004-llama-ext-h.patch \
    005-llama-cparams-h.patch \
    006-llama-hparams.patch \
    007-llama-arch.patch \
    008-llama-context.patch \
    009-llama-graph.patch \
    010-llama-memory.patch \
    011-llama-model.patch \
    012-llama-model-loader.patch \
    013-models-qwen35.patch \
    014-models-qwen35moe.patch \
    015-models-delta-net-base.patch \
    016-models-models-h.patch \
    017-models-qwen3next.patch \
    018-common-common-cpp.patch \
    019-common-arg.patch \
    020-common-download.patch \
    021-ggml-backend.patch \
    022-convert-hf-to-gguf.patch \
    023-server-context.patch; do
    if [ -f "$PATCH_DIR/$patch" ]; then
        echo "  Applying $patch..."
        patch -p1 < "$PATCH_DIR/$patch"
    else
        echo "  WARNING: $patch not found, skipping"
    fi
done
echo "All MTP patches applied."
```

---

## 5. Generazione delle patch

### Metodo raccomandato: da un fork di prova

1. Clonare `llama-cpp-turboquant` in `/tmp/mtp-base`
2. Clonare il branch `mtp-clean` di am17an in `/tmp/mtp-pr`
3. Estrarre il diff tra i due per ogni file del PR che esiste anche in turboquant:

```bash
# Per ogni file modificato dal PR che esiste in turboquant:
cd /tmp/mtp-base
diff -u src/llama-arch.cpp /tmp/mtp-pr/src/llama-arch.cpp > 007-llama-arch.patch
# ... ecc per ogni file
```

4. Per file nuovi (es. modelli MTP graph), creare patch che li aggiungono:

```bash
diff -u /dev/null /tmp/mtp-pr/src/models/models.h > 016-models-models-h.patch
```

### Verifiche sulle patch

Ogni patch deve:
- Applicarsi con `patch -p1` dalla root di `uninference/`
- NON modificare file già precedentemente modificati da altre patch nello stesso ordine
- Essere indipendente (se possibile) per facilitare skip/revert selettivi

---

## 6. Casi d'uso e parametri

### Scenario 1: Modello non-MTP (invariato) — **REGRESSION TEST**

```bash
# Stesso comando di prima, deve funzionare identicamente:
llama-server \
    --port 8080 \
    --host 0.0.0.0 \
    --models-dir /models \
    --models-max 1 \
    --cache-type-k turbo4 \
    --cache-type-v turbo3 \
    -c 260000 \
    -ngl 999 \
    --flash-attn on \
    --ubatch-size 256 \
    --batch-size 8192 \
    --threads 8 \
    --threads-batch 24
```

Il modello `unsloth/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (non-MTP) deve caricarsi e generare identicamente.

### Scenario 2: Modello MTP — **CON --spec-type mtp**

```bash
# Scaricare il GGUF MTP
# HF: unsloth/Qwen3.6-35B-A3B-MTP-GGUF
# Il GGUF contiene sia i trunk layers che l'MTP head

llama-server \
    --port 8080 \
    --host 0.0.0.0 \
    -m /models/Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf \
    --cache-type-k turbo4 \
    --cache-type-v turbo3 \
    -c 260000 \
    -ngl 999 \
    --flash-attn on \
    --ubatch-size 256 \
    --batch-size 8192 \
    --threads 8 \
    --threads-batch 24 \
    --spec-type mtp \
    --spec-draft-n-max 3
```

**Spiegazione parametri:**
- `--spec-type mtp` — attiva MTP speculative decoding (default: nessuno)
- `--spec-draft-n-max 3` — numero draft token da generare (default: 3, range consigliato 1-4)
- `--spec-draft-n-max 1` — minima latenza aggiuntiva, accettance rate ~88%
- `--spec-draft-n-max 3` — miglior bilanciamento latenza/throughput (~75% accettanza)
- `--spec-draft-n-max 4+` —吞吐 maggiore ma acceptance rate cala

### Scenario 3: Modello MTP via HF auto-download

```bash
llama-server \
    --port 8080 \
    -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF \
    --spec-type mtp \
    # ... altri parametri invariati
```

Il `--spec-type mtp` fa sì che il server cerchi automaticamente il file `mtp-*.gguf` nella stessa HF repo e lo carichi come MTP head.

### Scenario 4: MTP con specifica manuale del draft model

```bash
# Se il gguf MTP e il gguf trunk sono separati:
llama-server \
    --port 8080 \
    -m /models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
    -hfd unsloth/Qwen3.6-35B-A3B-MTP-GGUF \
    --spec-type mtp \
    --spec-draft-n-max 3
```

---

## 7. Piano di test per regressioni

### 7.1 Test automatici (da eseguire DOPO ogni patch group)

| Test | Comando | Verifica |
|------|---------|----------|
| Build | `cd uninference && mkdir -p build && cd build && cmake .. -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON && make -j$(nproc)` | Compilazione senza errori |
| Test arch | `./build/bin/llama-archs-test` | Tutti i modelli vengono riconosciuti |
| Test backend | `./build/bin/test-backend-ops` | GDN ops passano |
| Server avvio | `llama-server --version` | Versione corretta |

### 7.2 Test manuali su modello non-MTP

| Test | Comando | Verifica |
|------|---------|----------|
| Caricamento modello | Avvio server con GGUF non-MTP | `model loaded` senza errori MTP |
| Generazione | Richiesta chat completion | Output identico a prima (testa con seed fisso) |
| Cache KV | `--cache-type-k turbo4 --cache-type-v turbo3` | KV cache funziona |
| Flash attention | `--flash-attn on` | Nessun crash |

### 7.3 Test manuali su modello MTP

| Test | Comando | Verifica |
|------|---------|----------|
| Caricamento MTP | `--spec-type mtp` | Server carica MTP head |
| Generazione MTP | Richiesta con `--spec-draft-n-max 3` | Output sensato, `draft acceptance rate` > 60% |
| Speeedup | Confronto con e senza MTP | MTP deve essere più veloce |

---

## 8. Ordine di implementazione suggerito (dipendenze topologiche)

```
Fase 1: BASE (nessuna dipendenza)
  1. common/common.h         — enum + need_n_rs_seq()
  2. include/llama.h          — llama_context_type + n_rs_seq
  3. src/llama-cparams.h      — n_rs_seq + embeddings_pre_norm + ctx_type
  4. src/llama-hparams.h/cpp  — kv_only_nextn
  5. src/llama-graph.h        — LLM_GRAPH_TYPE_DECODER_MTP + t_h_pre_norm
  6. src/llama-ext.h          — staging API pre-norm

Fase 2: ARCHITETTURA
  7. src/llama-arch.cpp/h     — LAYER_REPEATING + llm_arch_supports_rs_rollback()

Fase 3: MEMORIA
  8. src/llama-memory.h       — ctx_type in params
  9. src/llama-memory-recurrent.*  — n_rs_seq
  10. src/llama-memory-hybrid.*    — propagate n_rs_seq

Fase 4: CONTEXTO CENTRALE
  11. src/llama-context.h/cpp — ctx_type_to_graph_type(), pre-norm, encode/decode changes

Fase 5: MODELLI
  12. src/models/delta-net-base.*  — keep_rs(), build_conv_state(), build_recurrent_attn()
  13. src/models/models.h          — graph_mtp dichiarazioni
  14. src/models/qwen35.cpp        — graph + graph_mtp + load_arch refactor
  15. src/models/qwen35moe.cpp     — graph + graph_mtp + load_arch refactor
  16. src/models/qwen3next.cpp     — refactor per build_conv_state()

Fase 6: GGML BACKEND
  17. ggml/src/ggml.c              — stato 3D GDN
  18. ggml/src/ggml-cpu/           — GDN snapshot CPU
  19. ggml/src/ggml-cuda/gated_delta_net.cu — GDN snapshot CUDA
  20. ggml/src/ggml-metal/         — GDN snapshot Metal
  21. ggml/src/ggml-vulkan/        — GDN snapshot Vulkan

Fase 7: MODEL LOADER
  22. src/llama-model-loader.*     — done_getting_tensors(partial)
  23. src/llama-model.cpp          — MTP contesto init + validazione

Fase 8: COMMON
  24. common/common.cpp            — seq_rm_type RS + n_rs_seq propagation
  25. common/speculative.h/cpp     — MTP speculative implementation
  26. common/arg.cpp               — MTP model download
  27. common/download.cpp/h        — MTP sibling discovery

Fase 9: SERVER + CONVERSION
  28. tools/server/server-context.cpp  — MTP init in server
  29. convert_hf_to_gguf.py            — MTP GGUF conversion
```

---

## 9. Rischi e mitigazioni

| Rischio | Impatto | Mitigazione |
|---------|---------|-------------|
| `build_rs()` cambiamento in `llama-graph.cpp` rompe altri modelli ricorrenti | ALTO | Testare con Mamba, RWKV, etc. Usare `s->ne[1]` corretto. |
| Refactor `build_layer_attn_linear()` in `delta-net-base.cpp` | ALTO | `keep_rs()` = false di default. Comportamento identico con `n_rs_seq == 0`. |
| Modifiche GDN backend (CUDA/Metal/Vulkan) per K>1 | ALTO | K==1 invariato. Testare su tutti i backend. |
| `encode()`/`decode()` modifiche in `llama-context.cpp` | ALTO | Tutte conditionali su `ctx_type`. Default invariato. |
| Patch non si applicano pulitamente su nuova versione di llama-cpp-turboquant | MEDIO | Le patch possono fallare (righe spostate). Bisognerà rigenerarle ad ogni update di llama-cpp-turboquant. Alternativa: mantenere un fork dedicato. |
| `LLM_TENSOR_LAYER_OUTPUT` → `LAYER_REPEATING` per NEXTN | MEDIO | GGUFs MTP hanno `blk.40.nextn.*`. GGUFs non-MTP non hanno NEXTN tensori. |

---

## 10. Raccomandazione finale

**Alternativa A (patch su update-llama.sh)** — Come descritto sopra. Manutenibile ma fragile
ad ogni update di llama-cpp-turboquant (le patch possono non applicarsi pulitamente).

**Alternativa B (fork dedicato)** — Mantenere un fork permanente di llama-cpp-turboquant
con MTP già mergiato. `update-llama.sh` punta al fork invece che all'originale.
Più stabile ma richiede manutenzione del fork.

**Alternativa C (aspettare merge upstream)** — PR #22673 è attivamente reviewata
(commenti da ggerganov e ngxson). Una volta mergiata in master di ggml-org/llama.cpp,
TheTom la incorporerà in llama-cpp-turboquant. A quel punto nessuna patch serve.

**Consiglio**: Iniziare con l'Alternativa A (patch) per prototipare subito, con l'opzione
di migrare all'Alternativa B se le patch diventano troppo fragili.
Monitorare lo stato della PR #22673 — se viene mergiata upstream, le patch diventano inutili.
