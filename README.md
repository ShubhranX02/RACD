# RACD — Retrieval-Augmented Circuit Design

Automated CTLE equalizer sizing via Reinforcement Learning, warm-started
by retrieval from a repository of previously-solved designs.

## Setup
See ProjectGuide.md Part 2.

## Run
```bash
cd src
python train.py                  # train and populate the repository
python compare_warmstart.py      # generate the headline warm-start-vs-random comparison
streamlit run dashboard.py       # launch the interactive demo
```

## Phase 8: Transistor-Level Extension (SKY130)
Ensure you have the PDK installed via `ciel` first:
```bash
ciel enable --pdk-family sky130 bdc9412b3e468c102d01b7cf6337be06ec6e9c9a
```
Then train and validate the transistor model:
```bash
python src/train_transistor.py   # trains on nominal TT corner for speed
python src/validate_pvt.py       # runs the candidate design across full PVT matrix
```
