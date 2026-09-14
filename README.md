# RACD — Retrieval-Augmented Circuit Design
**Autonomous RL-Driven High-Speed Link Equalization in Open-Source SkyWater 130 nm CMOS**

RACD integrates Reinforcement Learning (PPO), multi-objective FAISS vector retrieval, Ahead-of-Time ONNX compilation, and open-source EDA tools (`ngspice`, SkyWater 130 nm PDK) to automate the sizing, multi-corner PVT sign-off, and deployment of high-speed SerDes equalizers (PCIe Gen 2 / 5.0 Gbps).

---

## Key Results & Sign-Off

* **100% Multi-Corner PVT Sign-Off**: 5/5 corners passed (TT, SS, FF, SF, FS, $0\text{--}125^\circ\text{C}$, $V_{DD} \pm 5\%$) against strict PCIe Gen 2 specifications with zero human intervention.
* **Layout Budget Efficiency**: Synthesized 1-stage CTLE + 1-tap DFE layout area is **$0.00968\text{ mm}^2$** (**$80.6\%$ under** the $0.050\text{ mm}^2$ PHY limit).
* **Sample-Efficient Retrieval Warm-Starting**: Pre-training the policy via Behavioral Cloning over FAISS-retrieved designs outperforms cold random-initialization PPO on **$80\%$ of target specifications** (average $+0.689$ reward gain).
* **Ultra-Low Latency Inference**: Ahead-of-Time ONNX Runtime policy evaluates in **$20.2\ \mu\text{s}$** ($13.9\times$ faster than native PyTorch).
* **Continuous Tunability**: Full-spectrum verification from $3.0$ to $12.0\text{ dB}$ boost within $0.05\text{--}0.24\text{ dB}$ accuracy in milliseconds.

### Master 5-Corner PVT Sign-Off Table (Target: 8.0 dB)

| Corner | $V_{DD}$ | Temp | Peaking [$3\text{--}12\text{ dB}$] | HD3 [$< -30\text{ dB}$] | Noise [$< 1.5\text{ mV}$] | Power [$< 15\text{ mW}$] | Eye Height [$> 100\text{ mV}$] | Area [$< 0.05\text{ mm}^2$] | Compliance |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TT** | 1.80 V | 27°C | **7.76 dB** | **-31.8 dB** | **0.294 mVrms** | **3.39 mW** | **358.4 mV** | **0.00968 mm²** | **PASS [100%]** |
| **SS** | 1.71 V | 125°C | **6.67 dB** | **-32.0 dB** | **0.389 mVrms** | **3.20 mW** | **286.9 mV** | **0.00968 mm²** | **PASS [100%]** |
| **FF** | 1.89 V | 0°C | **8.25 dB** | **-35.5 dB** | **0.259 mVrms** | **3.57 mW** | **410.1 mV** | **0.00968 mm²** | **PASS [100%]** |
| **SF** | 1.80 V | 27°C | **7.95 dB** | **-32.7 dB** | **0.268 mVrms** | **3.39 mW** | **398.1 mV** | **0.00968 mm²** | **PASS [100%]** |
| **FS** | 1.80 V | 27°C | **7.50 dB** | **-31.7 dB** | **0.319 mVrms** | **3.39 mW** | **344.5 mV** | **0.00968 mm²** | **PASS [100%]** |

---

## Quickstart

### 1. Setup Environment
Ensure Python 3.12+ and `ngspice` are installed. Enable the SkyWater 130 nm PDK:
```bash
# Optional: install PDK via ciel if not already present
ciel enable --pdk-family sky130 bdc9412b3e468c102d01b7cf6337be06ec6e9c9a

# Install python dependencies
pip install -r requirements.txt
```

### 2. Run Verification & Tests
```bash
# Run unit test suite (20/20 tests passing)
pytest

# Run master end-to-end acceptance suite (Phases 1-9)
python src/verify_all.py

# Run master multi-corner PVT sign-off
python src/validate_pvt.py --target 8.0 --mode racd
```

### 3. Launch Interactive CAD Dashboard
```bash
streamlit run src/dashboard.py
```

### 4. Automated PDF Report Generation
```bash
python src/generate_report_pdf.py
# Generates reports/RACD_Final_Results_Report.pdf
```

---

## Repository Structure

```text
RACD2/
├── data/
│   ├── transistor_repository.jsonl      # Active SPICE-verified sizing repository
│   ├── pvt_validation_log.json          # Multi-corner PVT sign-off audit records
│   └── archive_2stage/                  # Preserved 2-stage exploration datasets
├── models/
│   ├── ppo_transistor.onnx              # Ahead-of-time compiled ONNX policy (20.2 us)
│   ├── ppo_transistor.zip               # Trained PPO actor-critic weights
│   ├── surrogate_transistor.pt          # 6-output PyTorch neural SPICE surrogate
│   ├── transistor_faiss.index           # FAISS nearest-neighbor vector database
│   └── archive_2stage/                  # Preserved 2-stage exploration models
├── reports/
│   ├── RACD_Final_Results_Report.pdf    # Publication-grade results summary
│   ├── chart_bode.png                   # SPICE ground-truth transfer curves
│   └── chart_warmstart.png              # Warm-start vs cold-init reward gains
├── src/
│   ├── circuit_transistor.py            # Sky130 ngspice in-memory simulation engine
│   ├── environment_transistor.py        # Gymnasium active equalizer MDP wrapper
│   ├── retrieval_faiss_transistor.py    # Sub-millisecond FAISS vector retrieval
│   ├── surrogate_transistor.py          # 6-output neural SPICE surrogate MLP
│   ├── train_transistor.py              # Zero-IPC vectorized PPO training pipeline
│   ├── validate_pvt.py                  # Master 5-corner PVT sign-off auditor
│   ├── orchestrator.py                  # Agentic natural language prompt parsing
│   ├── dashboard.py                     # Streamlit interactive CAD application
│   └── verify_all.py                    # Definition-of-Done master verification suite
├── tests/                               # 20 unit tests covering circuit, env, FAISS, surrogate
├── PROJECT_STATUS.md                    # Project completion & verification audit report
├── RACD_Final_Report.md                 # Comprehensive project final report
└── pytest.ini                           # Standard test discovery configuration
```
