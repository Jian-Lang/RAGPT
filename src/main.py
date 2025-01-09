from omegaconf import DictConfig
from utils import (
    seed_init, 
    generate_missing_table,
    Trainer)
import pandas as pd
import hydra
import warnings

warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig):
    pd.set_option('future.no_silent_downcasting', True)
    seed_init(cfg.seed)
    if cfg.regenerate_missing_table:
        generate_missing_table(**cfg.data_para)
    trainer = Trainer(cfg)
    trainer.run()

if  __name__ == '__main__':
    main()