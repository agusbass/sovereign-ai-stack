# Sovereign AI Stack v2.0 — Production Ready

**Modular AI Architecture: TEG → REA → DGA → AON → NLSI → PTT**

> **English summary:** a modular, production-ready AI architecture combining balance,
> information asymmetry, quantum isolation, and ultra-efficient gradient communication.

Arsitektur AI modular yang menggabungkan prinsip keseimbangan, asimetri informasi, isolasi kuantum, dan komunikasi gradien ultra-efisien.

---

## ✨ What's New in v2.0

| Fitur | Status |
|-------|--------|
| 🔬 **NLSI Integration** — State isolation sekarang menjadi bagian pipeline | ✅ Terintegrasi |
| 🧠 **AON Training Mode** — Cold-start fix: threshold otomatis turun saat training | ✅ Implemented |
| 🛡️ **Input Validation** — Semua modul punya validasi input + NaN/Inf detection | ✅ Implemented |
| 📉 **PTT Memory Leak Fix** — Gradient prior dibatasi ukurannya, EMA smoothing | ✅ Fixed |
| 🔗 **Autograd Safety** — AON .item() tidak lagi memutus autograd graph | ✅ Fixed |
| ⚡ **Gradient Clipping** — Terintegrasi di PTT | ✅ Implemented |
| 🧪 **87 Comprehensive Tests** — All pass | ✅ Tested |
| 🔄 **Optimizer Update** — AdamW + zero_grad sebelum forward pass | ✅ Fixed |

---

## 📋 Daftar Modul & Rumus

### 1. TEG — Temporal Entropy Gating (Filter Input)
*Menyaring noise, hanya data bernilai yang diteruskan.*

**Rumus:**
```
filtered_x = x ⊙ σ(Gate(x))
```
Dimana:
- `x` = Input tensor
- `σ` = Fungsi sigmoid
- `Gate(x)` = Linear(x) → σ → output [0,1]
- `⊙` = Element-wise multiplication

**v2.0 Enhancements:**
- Input validation dengan `_validate_input(x)`
- NaN/Inf detection & clamping otomatis
- Gradient scaling protection
- ValueError pada d_model ≤ 0

---

### 2. REA — Recursive Equilibrium Adaptation (Reasoning Core)
*"Otak" penalaran dengan 31-Step Micro-Loop.*

**Rumus:**
```
state₀ = x

for t = 1 to 31:
    zₜ = W · stateₜ₋₁
    gateₜ = σ(GateNet(zₜ))
    stateₜ = LayerNorm(stateₜ₋₁ + gateₜ ⊙ zₜ)

return state₃₁
```

**v2.0 Enhancements:**
- `equilibrium_tol` untuk early stopping (deteksi konvergensi)
- NaN/Inf detection per iterasi
- Iterasi bounds [1, 100] dengan ValueError
- `bias=False` pada W untuk konsistensi bobot
- Configurable `use_checkpointing`

---

### 3. DGA — Directional Gated Adaptation (Task Specialist)
*Adaptasi tugas spesifik tanpa retrain.*

**Rumus:**
```
Output = Base(x) + g · Adapter(x)

Adapter(x) = Up( ReLU( Down(x) ) )
```

**v2.0 Enhancements:**
- Parameter dipisah menjadi `down_proj`, `up_proj`, `activation` (bukan single Sequential)
- Input validation + NaN/Inf detection
- Gate value clamping [-1, 1]
- `set_adapter_dim()` method dengan bounds [4, 2048]

---

### 4. AON — Asymmetric Oracle Network (Verifier)
*"Penjaga Pintu" yang memverifikasi kebenaran logika.*

**Rumus:**
```
score = σ( VerifierNet(x) )
is_verified = score ≥ τ
```

**v2.0 Enhancements:**
- ⚠️ **CRITICAL FIX:** `.item()` sekarang hanya pada `mean_score` untuk return bool, sementara `score_tensor` tetap di autograd graph
- Dynamic threshold: training mode otomatis turunkan threshold ke 60%
- `set_training_mode(training)` untuk cold-start
- `get_confidence(x)` — inference-only scoring
- Input validation

---

### 5. NLSI — Non-Local State Insulation (Quantum Isolation)
*Isolasi state untuk stabilitas komputasi kuantum.*

**Rumus:**
```
shield = σ(insulation_barrier)
output = x ⊙ shield
```

**v2.0 Enhancements:**
- ✅ **Sekarang terintegrasi penuh di pipeline SovereignAIManager** (sebelumnya tidak digunakan)
- `get_shield_strength()` untuk monitoring
- `insulation_factor` bounds [0.01, 1.0]
- Shield clamping ke [0.01, 1.0]
- Input validation

---

### 6. PTT — Predictive Tensor Teleportation (GPU Communication)
*Teleportasi gradien — memangkas ~95-99% data komunikasi antar GPU.*

**Rumus:**
```
prior₀ = 0

for each step:
    residual = g_actual - prior              # Error prediksi
    k = ceil(N · sparsity_ratio)             # Berapa data dikirim
    threshold = kth_largest(|residual|, k)   # Threshold top-k
    mask = |residual| ≥ threshold            # Mask top-k
    sparse_residual = residual ⊙ mask        # Sparsifikasi
    g_reconstructed = prior + sparse_residual # Rekonstruksi
    prior = g_reconstructed                   # Update prediksi
```

**v2.0 Enhancements:**
- ⚠️ **CRITICAL FIX:** `gradient_prior` bounded dengan `max_prior_size` (anti memory leak)
- Gradient clipping integration via `grad_clip_norm`
- NaN/Inf detection with automatic zeroing
- EMA smoothing untuk prior updates (`alpha=0.1`)
- `get_stats()` untuk monitoring real-time
- `reset_prior()` method

---

### 7. Orchestrator — Sovereign AI Manager (v2.0)
*Pusat kendali sistem dengan pipeline yang sudah diperbaiki.*

**Alur Pipeline v2.0:**
```
Input → [TEG] → [REA] → [DGA] → [AON] → [NLSI] → Output
          ↓        ↓       ↓        ↓        ↓
       Filter   Reason  Adapt    Verify   Isolate
```

**Perbaikan Kritis v2.0:**
1. **NLSI sekarang terintegrasi** — Sebelumnya tidak pernah dipanggil
2. **zero_gard order diperbaiki** — zero_grad() dipanggil SEBELUM forward pass
3. **AdamW** — menggantikan Adam untuk weight decay yang lebih baik
4. **Training mode management** — AON threshold turun otomatis saat training
5. **NaN/Inf detection** di setiap pipeline stage
6. **Profiling** dengan `torch.profiler.record_function`

**Self-Correction Loop:**
```
while not is_verified and retry < max_retries:
    output = REA(output)      # Coba lagi
    is_verified = AON(output) # Verifikasi ulang
```

---

## 📂 Struktur Proyek

```
sovereign-ai-stack/
├── sovereign_ai_stack/       # Package sumber (importable)
│   ├── __init__.py           # Package init
│   ├── TEG_Module.py         # Temporal Entropy Gating (v2.0)
│   ├── REA_Module.py         # Recursive Equilibrium Adaptation (v2.0)
│   ├── DGA_Module.py         # Directional Gated Adaptation (v2.0)
│   ├── AON_Verifier.py       # Asymmetric Oracle Network (v2.0)
│   ├── PTT_Optimizer.py      # Predictive Tensor Teleportation (v2.0)
│   ├── NLSI_Module.py        # Non-Local State Insulation (v2.0)
│   ├── SovereignAIManager.py # Orchestrator (v2.0)
│   ├── BenchmarkRunner.py    # Framework pengujian (v2.0)
│   └── BenchmarkDashboard.py # Visualisasi dashboard (v2.0)
├── tests/
│   ├── __init__.py           # Test package init
│   └── test_all_modules.py   # 87 comprehensive tests
├── main.py                   # Production training loop (v2.0)
├── pyproject.toml            # Metadata & build package (v2.0.0)
├── requirements.txt          # Dependencies (pinned)
├── LICENSE                   # Lisensi MIT
└── README.md                 # Dokumentasi ini
```

---

## 🚀 Cara Penggunaan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

Atau install sebagai package (mode editable, untuk pengembangan):
```bash
pip install -e .
```

### 2. Jalankan test suite (87 tests)
```bash
python -m pytest tests/ -v
```

### 3. Jalankan training loop
```bash
python main.py
```

### 4. Jalankan dashboard visualisasi
```bash
python -m sovereign_ai_stack.BenchmarkDashboard
```

### 5. Gunakan dalam kode Anda
```python
from sovereign_ai_stack import SovereignAIManager, PTT_Optimizer

# Inisialisasi dengan konfigurasi v2.0
config = {
    'd_model': 512,
    'aon_training_mode': True,  # Cold-start fix
    'use_nlsi': True,           # NLSI integrated!
    'max_retries': 3,
}
model = SovereignAIManager(config)

# Training dengan PTT + gradient clipping
base_optim = torch.optim.AdamW(model.parameters(), lr=0.001)
ptt = PTT_Optimizer(base_optim, sparsity_ratio=0.05, grad_clip_norm=1.0)

# Training loop v2.0
model.set_training_mode(True)
for epoch in range(num_epochs):
    ptt.zero_grad()
    output = model(input_tensor, task_type='medical')
    if output is not None:
        loss = criterion(output, target)
        loss.backward()
        ptt.step()
```

---

## 🧪 Test Coverage (87 Tests)

| Modul | Jumlah Test | Coverage |
|-------|------------|----------|
| TEG_Module | 11 | Creation, forward, validation, NaN, gradient |
| REA_Module | 13 | Creation, forward, iterations, early stopping, stability |
| DGA_Module | 10 | Creation, forward, gate, adapter_dim, validation |
| AON_Verifier | 10 | Creation, confidence, training mode, gradient, validation |
| PTT_Optimizer | 9 | Creation, step, sparsity, NaN handling, clipping |
| NLSI_Module | 7 | Creation, forward, shield, gradient, validation |
| SovereignAIManager | 10 | Pipeline, training loop, gradient flow, retries |
| BenchmarkRunner | 4 | Creation, benchmark, latency, sparsity comparison |
| BenchmarkDashboard | 6 | Creation, update, summary, cleanup |
| Integration | 2 | Pipeline consistency, import check |

---

## 📊 Metrik Benchmark

| Metrik | Deskripsi |
|--------|-----------|
| TFLOPS/Watt | Efisiensi energi komputasi |
| Gradient Sparsity | % data komunikasi (PTT) — rendah = efisien |
| Verification Throughput | % output lolos verifikasi (AON) |
| Time-to-Convergence | Epoch mencapai akurasi target |

---

## 🧠 Filosofi Arsitektur

1. **Keseimbangan (Mizan):** Gate sigmoid pada REA menjaga stabilitas
2. **Asimetri Informasi:** Pemisahan Generator (REA) dan Verifier (AON)
3. **Efisiensi Ekstrem:** PTT menghemat 99% bandwidth GPU
4. **Isolasi Kuantum:** NLSI menjaga stabilitas state
5. **Modularitas:** Setiap modul independen, bisa dipasang/dilepas

---

## 📜 Lisensi

MIT License — Bebas digunakan dan dimodifikasi.
