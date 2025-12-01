import numpy as np
import torch.utils.data
import pandas as pd
import os

from PIL import Image

class Food101Dataset(torch.utils.data.Dataset):
    def __init__(self, split, max_text_len,  missing_type, missing_rate, **kargs):
        super().__init__()
        dataframe = pd.read_pickle(os.path.join('dataset/food101', f'{split}.pkl'))
        if missing_type == "image" or missing_type == "text":
            missing_table = pd.read_pickle('dataset/missing_table/single/food101/missing_table.pkl')
        elif missing_type == "both":
            missing_table = pd.read_pickle('dataset/missing_table/both/food101/missing_table.pkl')
        dataframe = pd.merge(dataframe, missing_table, on='item_id')
        self.missing_type = missing_type
        self.max_text_len = max_text_len
        self.id_list = dataframe['item_id'].tolist()
        self.text_list = dataframe['text'].tolist()
        self.label_list = dataframe['label'].tolist()
        self.missing_mask_list = dataframe[f'missing_mask_{int(10 * missing_rate)}'].tolist()

    def __getitem__(self, index):
        text = self.text_list[index]
        image = Image.open(fr'dataset/food101/image/{self.id_list[index]}.jpg').convert("RGB")
        if self.missing_type == "text" and self.missing_mask_list[index] == 0:
            text = ''
        elif self.missing_type == "both" and self.missing_mask_list[index] == 0:
            text = ''
        
        return {
            "image": image,
            "text": text,
            "label": self.label_list[index],
            "missing_mask": self.missing_mask_list[index],
        }
    def __len__(self):
        return len(self.text_list)