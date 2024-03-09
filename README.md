# Code for the Final Project: RNN-Transducer-based Losses for Speech Recognition on Noisy Targets

## Structure

```
.
├── README.md    # this file, readme
├── min_rnnt     # min-rnnt package, containing model and losses
│   ├── __init__.py
│   ├── decoding.py
│   ├── losses     # implementation of the proposed losses
│   │   ├── __init__.py
│   │   ├── bypass_transducer.py
│   │   ├── star_transducer.py
│   │   └── target_robust_transducer.py
│   ├── metrics.py  # WER metric for the model
│   ├── models.py   # RNN-T model (core code)
│   └── modules.py  # modules (Joint and Prediction Network)
├── notebooks
│   └── Loss_Demo.ipynb  # Jupyter Notebook with losses demo 
├── pyproject.toml  # project settings, formatting settings
├── requirements.txt  # project requirements
├── scripts
│   ├── conf
│   │   ├── conformer_transducer_med_min.yaml
│   │   └── fast_conformer_transducer_min.yaml
│   ├── generate_data.py   # script to generate corrupted data
│   └── train_min_rnnt.py  # training script
├── setup.py
└── tests  # unit tests
    ├── test_bypass_transducer.py
    ├── test_metrics.py
    ├── test_star_transducer.py
    └── test_target_robust_transducer.py
```

## Reproducing experiments

- generate 