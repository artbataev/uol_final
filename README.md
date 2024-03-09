# Code for the Final Project: RNN-Transducer-based Losses for Speech Recognition on Noisy Targets

The losses are planned to be proposed to the [NeMo](https://github.com/nvidia/nemo) framework.

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
├── pyproject.toml   # project settings, formatting settings
├── requirements.txt  # project requirements
├── scripts
│   ├── conf
│   │   ├── conformer_transducer_med_min.yaml   # Conformer-Medium config
│   │   └── fast_conformer_transducer_min.yaml  # Fast Conformer config
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

- clone the repository
- install dependencies `pip install -r requirements.txt`
- install the package `min_rnnt`: `pip install .`
- download and prepare [LibriSpeech](https://www.openslr.org/12) data (including "clean" manifests) using the script from the [NeMo](https://github.com/nvidia/nemo) framework `https://github.com/NVIDIA/NeMo/blob/v1.21.0/scripts/dataset_processing/get_librispeech_data.py`
- use script to generate corrupted data as described in the report, e.g., `python generate_data.py --src-path=train_manifest.jsonl --dst-path=train_manifest_del0.5.jsonl --del-prob=0.5`; for parameters see the help `python generate_data.py --help`
- create a BPE tokenizer with 1024 units using training manifests and script `https://github.com/NVIDIA/NeMo/blob/v1.21.0/scripts/tokenizers/process_asr_text_tokenizer.py`
- train the model `python scripts/train_min_rnnt.py --config-name=fast_conformer_transducer_min --config-path=./scripts/conf odel.tokenizer.dir=<tokenizer_dir> model.loss.loss_name="<rnnt, star_t, bypass_t or trt> model.train_ds.manifest_filepath=training_manifest_del0.5.jsonl model.validation_ds.manifest_filepath=dev_other.jsonl"`
