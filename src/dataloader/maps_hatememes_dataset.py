import numpy as np
import torch.utils.data
import os
import pandas as pd
from PIL import Image

class HatememesDataset(torch.utils.data.Dataset):
    def __init__(self, split, max_text_len,  missing_type, missing_rate, **kargs):
        super().__init__()
        dataframe = pd.read_pickle(os.path.join('dataset/hatememes', f'{split}.pkl'))
        if missing_type == "image" or missing_type == "text":
            missing_table = pd.read_pickle('dataset/missing_table/single/hatememes/missing_table.pkl')
        elif missing_type == "both":
            missing_table = pd.read_pickle('dataset/missing_table/both/hatememes/missing_table.pkl')
        dataframe = pd.merge(dataframe, missing_table, on='item_id')
        self.missing_type = missing_type
        self.max_text_len = max_text_len
        self.id_list = dataframe['item_id'].tolist()
        self.text_list = dataframe['text'].tolist()
        self.label_list = dataframe['label'].tolist()
        self.missing_mask_list = dataframe[f'missing_mask_{int(10 * missing_rate)}'].tolist()

    def __getitem__(self, index):
        # k = self.k
        text = self.text_list[index]
        image = Image.open(fr'dataset/hatememes/image/{self.id_list[index]}.png').convert("RGB")
        
        return {
            "image": image,
            "text": text,
            "label": self.label_list[index],
            "missing_mask": self.missing_mask_list[index],
        }
    def __len__(self):
        return len(self.text_list)