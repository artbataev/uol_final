# This is the part of the Final Project
# "RNN-Transducer-based Losses for Speech Recognition on Noisy Targets"
# Originally published in https://github.com/artbataev/uol_final

import pytorch_lightning as pl
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf

from min_rnnt.models import MinRNNTModel

# The training script with NeMo, derived from standard traning script
# https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_transducer/speech_to_text_rnnt.py
# the only customization is using MinRNNTModel


@hydra_runner(config_path="conf", config_name="fast_conformer_transducer_min")
def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    # instantiate trainer
    trainer = pl.Trainer(**cfg.trainer)
    # use NeMo experiment manager (with WandB logging)
    exp_manager(trainer, cfg.get("exp_manager", None))
    # instantiate model
    rnnt_model = MinRNNTModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    rnnt_model.maybe_init_from_pretrained_checkpoint(cfg)

    # train the model (+ validation after each epoch)
    trainer.fit(rnnt_model)

    # test the model if necessary
    if hasattr(cfg.model, "test_ds") and cfg.model.test_ds.manifest_filepath is not None:
        if rnnt_model.prepare_test(trainer):
            trainer.test(rnnt_model)


if __name__ == "__main__":
    main()
