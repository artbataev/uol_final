import pytorch_lightning as pl
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf

from min_rnnt.models import MinRNNTModel


@hydra_runner(config_path="conf", config_name="fast_conformer_transducer_min")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    rnnt_model = MinRNNTModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    rnnt_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(rnnt_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if rnnt_model.prepare_test(trainer):
            trainer.test(rnnt_model)


if __name__ == '__main__':
    main()
