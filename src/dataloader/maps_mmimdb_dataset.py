import numpy as np
import torch.utils.data
import os
import pandas as pd
from PIL import Image


class MMIMDbDataset(torch.utils.data.Dataset):
    def __init__(self, split, max_text_len,  missing_type, missing_rate, **kargs):
        super().__init__()
        dataframe = pd.read_pickle(os.path.join('dataset/mmimdb', f'{split}.pkl'))
        if missing_type == "image" or missing_type == "text":
            missing_table = pd.read_pickle('dataset/missing_table/single/mmimdb/missing_table.pkl')
        elif missing_type == "both":
            missing_table = pd.read_pickle('dataset/missing_table/both/mmimdb/missing_table.pkl')
        dataframe = pd.merge(dataframe, missing_table, on='item_id')
        self.missing_type = missing_type
        self.max_text_len = max_text_len
        self.id_list = dataframe['item_id'].tolist()
        self.text_list = dataframe['text'].tolist()
        self.label_list = dataframe['label'].tolist()
        self.missing_mask_list = dataframe[f'missing_mask_{int(10 * missing_rate)}'].tolist()

    def __getitem__(self, index):
        text = self.text_list[index]
        image = Image.open(fr'dataset/mmimdb/image/{self.id_list[index]}.jpeg').convert("RGB")
        # 把下面的image的任何处理取消，改在collator里面把baseline的image tensor做一个置空，然后改一下那个初始化，加一个nn.parameter，之后就一比一还原了。
        '''
         # missing image, dummy image is all-one image
        if self.missing_table[image_index] == 2 or simulate_missing_type == 2:
            for idx in range(len(image_tensor)):
                image_tensor[idx] = torch.ones(image_tensor[idx].size()).float()
            
        '''
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