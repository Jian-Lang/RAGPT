import numpy as np
import torch.utils.data
import os
import pandas as pd
from PIL import Image

class HatememesDataset(torch.utils.data.Dataset):
    def __init__(self, split, max_text_len,  missing_type, missing_rate, k, **kargs):
        super().__init__()
        dataframe = pd.read_pickle(os.path.join('dataset/hatememes', f'{split}.pkl'))
        if missing_type == "image" or missing_type == "text":
            missing_table = pd.read_pickle('dataset/missing_table/single/hatememes/missing_table.pkl')
        elif missing_type == "both":
            missing_table = pd.read_pickle('dataset/missing_table/both/hatememes/missing_table.pkl')
        dataframe = pd.merge(dataframe, missing_table, on='item_id')
        self.k = k
        self.missing_type = missing_type
        self.max_text_len = max_text_len
        self.id_list = dataframe['item_id'].tolist()
        self.text_list = dataframe['text'].tolist()
        self.label_list = dataframe['label'].tolist()
        self.i2i_list = dataframe['i2i_id_list'].tolist()
        self.t2t_list = dataframe['t2t_id_list'].tolist()
        self.i2i_r_l_list_list = dataframe['i2i_label_list'].tolist()
        self.t2t_r_l_list_list = dataframe['t2t_label_list'].tolist()
        self.missing_mask_list = dataframe[f'missing_mask_{int(10 * missing_rate)}'].tolist()

    def __getitem__(self, index):
        k = self.k
        text = self.text_list[index]
        image = Image.open(fr'dataset/hatememes/image/{self.id_list[index]}.png').convert("RGB")
        r_t_list = []
        r_i_list = []

        if self.missing_type == "text" and self.missing_mask_list[index] == 0:
            text = "I love deep learning" * 1024
            i2i_list = self.i2i_list[index]

            for i in i2i_list[:k]:
                r_i = np.load(fr'dataset/memory_bank/hatememes/image/{i}.npy')
                r_i_list.append(r_i.tolist())
                r_t = np.load(fr'dataset/memory_bank/hatememes/text/{i}.npy')
                r_t_list.append(r_t.tolist())

            r_l_list = self.i2i_r_l_list_list[index]

        elif self.missing_type == "image" and self.missing_mask_list[index] == 0:
            t2t_list = self.t2t_list[index]

            for i in t2t_list[:k]:
                r_t = np.load(fr'dataset/memory_bank/hatememes/text/{i}.npy')
                r_t_list.append(r_t.tolist())
                r_i = np.load(fr'dataset/memory_bank/hatememes/image/{i}.npy')
                r_i_list.append(r_i.tolist())
            r_l_list = self.t2t_r_l_list_list[index]
            
        elif self.missing_type == "text" or self.missing_type == "image" and self.missing_mask_list[index] == 1:
            i2i_list = self.i2i_list[index]
            t2t_list = self.t2t_list[index]

            for i in i2i_list[:k]:
                r_i = np.load(fr'dataset/memory_bank/hatememes/image/{i}.npy')
                r_i_list.append(r_i.tolist())
            
            for i in t2t_list[:k]:
                r_t = np.load(fr'dataset/memory_bank/hatememes/text/{i}.npy')
                r_t_list.append(r_t.tolist())
            r_l_list = self.i2i_r_l_list_list[index]
        
        elif self.missing_type == "both" and self.missing_mask_list[index] == 0:
            text = "I love deep learning" * 1024
            i2i_list = self.i2i_list[index]
            for i in i2i_list[:k]:
                r_i = np.load(fr'dataset/memory_bank/hatememes/image/{i}.npy')
                r_i_list.append(r_i.tolist())
                r_t = np.load(fr'dataset/memory_bank/hatememes/text/{i}.npy')
                r_t_list.append(r_t.tolist())
            r_l_list = self.i2i_r_l_list_list[index]
        
        elif self.missing_type == "both" and self.missing_mask_list[index] == 1:
            t2t_list = self.t2t_list[index]
            for i in t2t_list[:k]:
                r_t = np.load(fr'dataset/memory_bank/hatememes/text/{i}.npy')
                r_t_list.append(r_t.tolist())
                r_i = np.load(fr'dataset/memory_bank/hatememes/image/{i}.npy')
                r_i_list.append(r_i.tolist())
            r_l_list = self.t2t_r_l_list_list[index]
        
        elif self.missing_type == "both" and self.missing_mask_list[index] == 2:
            i2i_list = self.i2i_list[index]
            t2t_list = self.t2t_list[index]

            for i in i2i_list[:k]:
                r_i = np.load(fr'dataset/memory_bank/hatememes/image/{i}.npy')
                r_i_list.append(r_i.tolist())
            
            for i in t2t_list[:k]:
                r_t = np.load(fr'dataset/memory_bank/hatememes/text/{i}.npy')
                r_t_list.append(r_t.tolist())
            
            r_l_list = self.i2i_r_l_list_list[index]
        
        return {
            "image": image,
            "text": text,
            "label": self.label_list[index],
            "r_t_list": r_t_list,
            "r_i_list": r_i_list,
            "missing_mask": self.missing_mask_list[index],
            "r_l_list": r_l_list[:k]
        }
    def __len__(self):
        return len(self.text_list)