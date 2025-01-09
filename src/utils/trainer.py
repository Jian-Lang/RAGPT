import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from colorama import Back, Fore, Style
from .core_tools import (
    get_optim, 
    make_saving_folder_and_logger, 
    print_init_msg,
    load_model,
    get_dataset,
    compute_loss,
    get_evaluator,
    Collator,
    EarlyStopping
)
import os
from omegaconf import DictConfig
import numpy as np
from omegaconf import DictConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Trainer():
    def __init__(self,
                 cfg: DictConfig):
        self.cfg = cfg
        self.epochs = cfg.epochs
        self.dataset = cfg.data_para.dataset
        self.device = cfg.device
        self.missing_type = cfg.data_para.missing_type
        self.task = cfg.data_para.dataset
        self.father_folder_name, self.folder_name, self.logger = make_saving_folder_and_logger(cfg)
        self.model = load_model(
            missing_type=cfg.data_para.missing_type, 
            task_id = self.task, 
            device = cfg.device, 
            max_text_len = cfg.data_para.max_text_len,
            max_image_len = cfg.data_para.max_image_len,
            **dict(cfg.model_para))
        self.model.to(self.device)
        train_dataset = get_dataset(dataset_name=self.dataset, split='train', **cfg.data_para)
        test_dataset = get_dataset(dataset_name=self.dataset, split='test', **cfg.data_para)
        collator = Collator(max_text_len=cfg.data_para.max_text_len)
        self.train_data_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, collate_fn=collator, num_workers=cfg.num_workers, shuffle=True)
        self.test_data_loader = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, collate_fn=collator, num_workers=cfg.num_workers)
        if self.dataset != 'food101':
            valid_dataset = get_dataset(dataset_name=self.dataset, split='valid', **cfg.data_para)
            self.valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=cfg.batch_size, collate_fn=collator, num_workers=cfg.num_workers, shuffle=True)
        self.optimizer, self.scheduler = get_optim(max_steps=len(self.train_data_loader) * self.epochs, model=self.model, **cfg.optim_para)
        self.evaluator = get_evaluator(self.dataset, cfg.device)
        self.early_stopping = EarlyStopping(patience=cfg.patience, path=os.path.join(self.father_folder_name, self.folder_name, 'best_model.pth'))
    def run(self):
        print_init_msg(self.logger, self.cfg)
        for epoch in range(self.epochs):
            self.logger.info(f'Current Epoch: {epoch + 1}')
            self._train()
            if self.dataset != 'food101':
                self._valid(split='valid', use_early_stopping=True)
                self._valid('test')
            else:
                self._valid('test', use_early_stopping=True)
            if self.early_stopping.early_stop:
                self.logger.info(f"{Fore.GREEN}Early stopping at epoch {epoch}")
                break
        self.model.load_state_dict(torch.load(os.path.join(self.father_folder_name, self.folder_name, 'best_model.pth'), weights_only=False))
        best_metrics = self._valid(split='test', final_turn=True)
        if self.task == 'hatememes':
            self.logger.info(f"Final Tesing AUROC: {best_metrics['auroc']}")
        elif self.task == 'food101':
            self.logger.info(f"Final Tesing Accuracy: {best_metrics['accuracy']}")
        elif self.task == 'mmimdb':
            self.logger.info(f"Final Tesing F1_micro: {best_metrics['f1_micro']}")

    def _train(self):
        loss_list =  []
        self.model.train()
        pbar = tqdm(self.train_data_loader, bar_format=f"{Fore.BLUE}{{l_bar}}{{bar}}{{r_bar}}", desc='Training')
        for batch in pbar:
            inputs = {key: val.to(self.device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}
            labels = inputs.pop('label')
            preds = self.model(**inputs)
            loss = compute_loss(preds, labels, self.dataset)
            loss_list.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        self.logger.info(f"{Fore.BLUE}Train: Loss: {np.mean(loss_list)}{Style.RESET_ALL}")

    def _valid(self, split, use_early_stopping=False, final_turn=False):
        self.model.eval()
        loss_list = []

        if split == 'valid':
            f_color = Fore.YELLOW
            data_loader = self.valid_data_loader
            desc = 'Validating'
        elif split == 'test':
            f_color = Fore.RED
            data_loader = self.test_data_loader
            desc = 'Testing'
        else:
            raise ValueError(f"Invalid split: {split}")
        
        pbar = tqdm(data_loader, bar_format=f"{f_color}{{l_bar}}{{bar}}{{r_bar}}", desc=desc)

        with torch.no_grad():
            for batch in pbar:
                inputs = {key: val.to(self.device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}
                labels = inputs.pop('label')
                preds = self.model(**inputs)
                loss = compute_loss(preds, labels, self.dataset)
                loss_list.append(loss.item())
                self.evaluator.update(preds=preds, labels=labels)
    
        metrics = self.evaluator.compute()
        if not final_turn:
            self.logger.info(f"{f_color}{split}: Loss: {np.mean(loss_list)}{Style.RESET_ALL}")
            if self.task == 'hatememes':
                self.logger.info(f"{f_color}{split}: AUROC: {metrics['auroc']}{Style.RESET_ALL}")
                score = metrics['auroc']
            elif self.task == 'food101':
                self.logger.info(f"{f_color}{split}: Accuracy: {metrics['accuracy']}{Style.RESET_ALL}")
                score = metrics['accuracy']
            elif self.task == 'mmimdb':
                self.logger.info(f"{f_color}{split}: F1_micro: {metrics['f1_micro']}{Style.RESET_ALL}")
                self.logger.info(f"{f_color}{split}: F1_sample: {metrics['f1_sample']}{Style.RESET_ALL}")
                score = metrics['f1_micro']
            else:
                raise ValueError(f"Invalid task: {self.task}")
            if use_early_stopping:
                self.early_stopping(score, self.model)
        else:
            return metrics