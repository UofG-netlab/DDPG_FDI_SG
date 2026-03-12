# DDPG_FDI_SG
Enhancing Smart Grid Cyber Resilience against FDI attacks using Multi-Agent Recurrent DDPG

Contributions: Dr. Tahira Mahboob (Netlab - UofG): tahira.mahboob@yahoo.com, Mingwei Li (UoE): ml969@exeter.ac.uk

## PandaPower (Pandapower + RL)

This repository contains:
- Legacy training/evaluation scripts based on `pandapower.timeseries.run_timeseries` and custom controllers.
- A PettingZoo/RLlib path (multi-agent) for RL training.

### Setup

Create and activate venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Legacy scripts (run_timeseries-based)

Run DDPG training:

```bash
python tests/ddpg_training.py
```

Run LSTM-DDPG training:

```bash
python tests/ddpg_lstm/ddpg_lstm_training.py
```

Run DDPG evaluation (loads saved actor checkpoints):

```bash
python tests/ddpg_test.py
```

Run LSTM-DDPG evaluation:

```bash
python tests/ddpg_lstm/ddpg_lstm_test.py
```

### PettingZoo env smoke test

```bash
python tests/rllib_smoke_test.py
```


