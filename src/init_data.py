from omegaconf import DictConfig
from utils import (
    init_data_mmimdb,
    init_data_hatememes,
    init_data_food101,
    MemoryBankGenerator,
    MCR)
import pandas as pd
import hydra
import warnings

warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig):
    pd.set_option('future.no_silent_downcasting', True)
    print('==> Data Initialization start.')
    if cfg.data_para.dataset == 'mmimdb':
        init_data_mmimdb()
    elif cfg.data_para.dataset == 'hatememes':
        init_data_hatememes()
    elif cfg.data_para.dataset == 'food101':
        init_data_food101()
    print('==> Data Initialization finished.')
    print('==> Memory Bank Generation start.')
    memory_bank_generator = MemoryBankGenerator(cfg)
    memory_bank_generator.run()
    print('==> Memory Bank Generation finished.')
    print('==> Multi-Channel Retrieval start.')
    mcr = MCR(cfg)
    mcr.run()
    print('==> Multi-Channel Retrieval finished.')

if  __name__ == '__main__':
    main()