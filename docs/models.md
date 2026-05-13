# Modelli MoE per turboquant

Analisi dei modelli Mixture of Experts (MoE) compatibili con l'hardware
di riferimento e configurabili in router mode su `llama-server`.

---

## Hardware di riferimento

| Risorsa | Disponibile | Usabile (con overhead) |
|---------|-------------|------------------------|
| GPU VRAM | 8 GB | ~7 GB |
| RAM | 20 GB | ~17 GB |
| Storage modelli | mounts `models/` via `--models-dir /models` |

## Configurazione turboquant

- `-ngl 999` — offload massimo layer su GPU
- `--cache-type-k turbo4` — K cache a 4-bit (0.5 byte/elem)
- `--cache-type-v turbo3` — V cache a 3-bit (~0.375 byte/elem)
- `--no-mmap --mlock` — modello caricato interamente in RAM, locked
- `-c 131072` — contesto preallocato 128K token

---

## Modello attuale: Qwen3.6-35B-A3B-UD-Q4_K_M

Il modello in uso (`turboquant.sh` linea 2).

### Architettura

- **Params:** 34.7B totali / ~3.3B attivi
- **Layer:** 40 (10 blocchi × 3 GatedDeltaNet + 1 GatedAttention)
- **Attenzione:** ibrida — 30 layer lineari (GatedDeltaNet, nessuna KV cache) + 10 full attention
- **Full attention:** 16 Q-heads, 2 KV-heads, head-dim 128
- **Esperti:** 256 routed (top-8) + 1 shared, intermediate size 512
- **Vocab:** 248.320
- **Multimodale:** sì (vision encoder + text)
- **Licenza:** Apache 2.0
- **HF:** `unsloth/Qwen3.6-35B-A3B-GGUF`

### Memoria e contesto

| Parametro | Valore |
|-----------|--------|
| File Q4_K_M | ~19.5 GB |
| Layer offloadabili GPU | ~14/40 |
| KV cache/token (K4/V3, soli 10 layer attention) | ~2.2 KB |
| KV cache @ 128K | ~275 MB |
| KV cache @ 262K (nativa) | ~560 MB |
| KV cache @ 1M (max teorico) | ~2.2 GB |
| GPU VRAM @ 128K | ~7.6 GB ✓ |
| GPU VRAM @ 262K | ~7.8 GB ✓ |
| RAM @ 128K | ~12.6 GB ✓ |

### Vantaggi chiave

L'architettura ibrida è il punto di forza: 30 layer su 40 non hanno KV cache
(lineare, ricorrente). Questo rende la memoria per il contesto trascurabile
anche a 262K token, un risultato impossibile per un trasformer puro.

---

## Candidati per router mode

Tutti i modelli vanno scaricati come `.gguf` (Q4_K_M) e copiati in `models/`.
`llama-server` in router mode li serve automaticamente.
I client selezionano il modello via campo `"model"` nelle richieste.

### Qwen3-30B-A3B — `unsloth/Qwen3-30B-A3B-GGUF`

| Campo | Valore |
|-------|--------|
| **Params** | 30.5B totali / 3.3B attivi |
| **File Q4_K_M** | ~17.2 GB |
| **Layer** | 48 |
| **Attenzione** | GQA: 32 Q-heads, 4 KV-heads, head-dim 64 |
| **Esperti** | 128 routed (top-8), nessun shared |
| **Contesto nativo** | 32K (estendibile a 128K via YaRN) |
| **Vocab** | 151.936 |
| **Multimodale** | No |
| **Licenza** | Apache 2.0 |
| **HF** | `unsloth/Qwen3-30B-A3B-GGUF` |

#### Memoria

| Parametro | Valore |
|-----------|--------|
| KV cache/token (K4/V3) | ~10.5 KB |
| KV cache @ 32K | ~336 MB |
| KV cache @ 64K | ~672 MB |
| KV cache @ 128K | ~1.34 GB |
| Layer offloadabili GPU | ~19/48 |
| GPU VRAM @ 64K | ~7.9 GB ✓ |
| GPU VRAM @ 128K | ~8.5 GB ⚠️ (cache split VRAM/RAM) |
| RAM @ 64K | ~10.5 GB ✓ |

#### Contesto sicuro

| Contesto | Giudizio |
|----------|----------|
| 32K | ✅ Nativo, tutto in VRAM |
| 64K | ✅ Consigliato, cache in VRAM |
| 128K | ⚠️ Fattibile ma cache in split GPU/CPU |

#### Varianti disponibili (stessa impronta)

| Modello | Contesto | Link |
|---------|----------|------|
| Qwen3-30B-A3B-Instruct | 32K (128K max) | `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` |
| Qwen3-30B-A3B-128K | 128K nativo | `unsloth/Qwen3-30B-A3B-128K-GGUF` |
| Qwen3-30B-A3B-Thinking | 32K (128K max) | `unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF` |
| **Qwen3-Coder-30B-A3B** | **32K (128K max)** | **`unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`** |
| Qwen3-Coder-30B-A3B-1M | 1M max | `unsloth/Qwen3-Coder-30B-A3B-Instruct-1M-GGUF` |

La variante **1M** sul tuo HW è impraticabile a lunghezze >300K
(KV cache a 1M = ~10.5 GB).

---

### DeepSeek-Coder-V2-Lite-Instruct — `bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF`

| Campo | Valore |
|-------|--------|
| **Params** | 16B totali / ~2.4B attivi |
| **File Q4_K_M** | ~9 GB |
| **Layer** | 27 |
| **Attenzione** | MLA (Multi-head Latent Attention, kv_lora_rank=512) |
| **Esperti** | 64 routed (top-6) + 2 shared |
| **Contesto nativo** | 128K (via YaRN su base 4K) |
| **Vocab** | 102.400 |
| **Multimodale** | No |
| **Licenza** | DeepSeek License (permissiva per uso) |
| **HF** | `bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF` |

#### Memoria

| Parametro | Valore |
|-----------|--------|
| KV cache/token (K4/V3, MLA) | ~12 KB |
| KV cache @ 32K | ~380 MB |
| KV cache @ 128K | ~1.5 GB |
| Layer offloadabili GPU | ~21/27 (~7 GB su 9 totali) |
| GPU VRAM @ 128K | ~7.5 GB ✓ |
| RAM @ 128K | ~2.5 GB ✓ |

#### Contesto sicuro

| Contesto | Giudizio |
|----------|----------|
| 32K | ✅ Ampio margine |
| 64K | ✅ |
| 128K | ✅ Nativo, tutto headroom |

**Vantaggio:** modello molto più leggero (9 GB vs 19.5 GB). Incassa in GPU
quasi interamente. La latenza di inferenza sarà molto più bassa.
Ottimizzato per coding (DeepSeekCoder).

---

### DeepSeek-V2-Lite-Chat — `mradermacher/DeepSeek-V2-Lite-Chat-GGUF`

Stessa architettura MLA + MoE del Coder-V2-Lite, ma:

| Parametro | Valore |
|-----------|--------|
| Contesto nativo | 32K (non 128K) |
| Qualità | Inferiore (modello più vecchio) |

Sconsigliato in presenza del Coder-V2-Lite.

---

### Qwen3.5-35B-A3B — `unsloth/Qwen3.5-35B-A3B-GGUF`

Architettura identica al Qwen3.6-35B-A3B. Stessa impronta di memoria,
stesso contesto. Qwen3.6 è marginalmente superiore (post-training RL +
agentic coding). Non è un upgrade.

---

## Tabella comparativa

| Modello | File | Attivi | Ctx sicuro | KV@ctx_max | Licenza | unsloth |
|---------|------|--------|------------|------------|---------|---------|
| **Qwen3.6-35B-A3B** *(tuo)* | ~19.5 GB | 3.3B | **128K-262K** | 275-560 MB | Apache 2.0 | ✅ |
| **Qwen3-30B-A3B** | ~17.2 GB | 3.3B | **32K-64K** | 336-672 MB | Apache 2.0 | ✅ |
| **Qwen3-Coder-30B-A3B** | ~17.2 GB | 3.3B | **32K-64K** | 336-672 MB | Apache 2.0 | ✅ |
| **DS-Coder-V2-Lite** | ~9 GB | 2.4B | **128K** | 380-1500 MB | Permissiva | ❌ |
| **DS-V2-Lite-Chat** | ~9 GB | 2.4B | **32K** | 100-380 MB | Permissiva | ❌ |

## Raccomandazioni

### Primario (gia configurato)
**Qwen3.6-35B-A3B** — Tienilo come modello principale.
È già il miglior compromesso per 8 GB VRAM + 20 GB RAM.
Nessun altro modello nella stessa fascia di parametri offre:

- 262K contesto nativo
- KV cache quasi nulla grazie all'architettura ibrida
- Supporto multimodale
- Qualità comparabile per uso generale

### Secondario per coding (router mode)
Scegli uno tra:

1. **Qwen3-Coder-30B-A3B** (Apache 2.0, unsloth, ~17 GB)
   - Contesto 64K sicuro, 128K tight ma fattibile
   - Stessa famiglia del modello primario

2. **DeepSeek-Coder-V2-Lite-Instruct** (~9 GB)
   - Molto più leggero, massimo headroom
   - 128K nativo
   - Coding puro, ottime performance
   - Latenza inferiore

### Installazione

```bash
# Scarica il .gguf e mettilo in models/
cd models
# Esempio per Qwen3-Coder
wget https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf

# Oppure per DeepSeek
wget https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf
```

### Uso in router mode

```bash
# Lista modelli disponibili
curl http://localhost:8080/models

# Chat con modello specifico
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "scrivi una funzione che..."}]
  }'
```

---

## Modelli MoE non adatti al tuo HW

| Modello | Motivo |
|---------|--------|
| DeepSeek-V3 / V3.1 / V3.2 / V3.2.1 | 671B totali, troppo grande anche a Q4_K_M (~380 GB) |
| Mixtral 8x7B | 47B totali, ~27 GB a Q4_K_M, eccede budget |
| DBRX | 132B totali, troppo grande |
| JetMoE 8B | GGUF su HF morti / vuoti |
| OLMoE 1B-7B | Sotto la soglia minima (~6 GB a Q4) |
| Cogito-v2-preview-109B | 109B totali, troppo grande |

---

## Note sul calcolo della KV cache

Formula usata per la stima del contesto massimo:

```
K cache (turbo4) = layer_FULL × kv_heads × head_dim × 0.5 byte
V cache (turbo3) = layer_FULL × kv_heads × head_dim × 0.375 byte
KV totale/token  = K + V

VRAM usata = min(model_q4km, 7 GB) + KV_cache + 200 MB overhead
RAM usata  = model_q4km - offload_gpu + KV_cache_remainder + OS (~3 GB)
```

Dove `layer_FULL` sono solo i layer con attenzione full (non lineari).

Per **MLA**: i valori sono approssimativi perché il formato compresso
ha una struttura diversa dalla KV cache standard. I numeri sopra
assumono che la cache MLA sia quantizzabile come K4/V3, il che dipende
dal supporto in `llama.cpp`.
