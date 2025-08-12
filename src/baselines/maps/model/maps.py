# -*- coding: utf-8 -*-
# @Time    : 2024/7/14 下午3:16
# @Author  : jian lang
# @File    : maps.py
# @Description:

import torch
from PIL import Image
from torch import nn
from transformers import BertTokenizer

from baselines.maps.vilt import ViltModel, ViltImageProcessor


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class maps(torch.nn.Module):
    def __init__(self,
                 vilt: ViltModel,
                 dataset_name: str,
                 max_text_len: int,
                 missing_type: str,
                 learnt_p: bool,
                 prompt_type: str,
                 prompt_num: int,
                 prompt_len: int,
                 device: torch.device,
                 embed_dim=768,
                 **kargs):
        super(maps, self).__init__()
        self.device = device
        self.dataset_name = dataset_name
        self.max_text_len = max_text_len
        self.learnt_p = learnt_p
        self.prompt_type = prompt_type
        self.prompt_len = prompt_len
        self.prompt_num = prompt_num
        self.total_prompt_len = prompt_len * prompt_num
        if dataset_name == "hatememes":
            cls_num = 2
        elif dataset_name == "food101":
            cls_num = 101
        elif dataset_name == "mmimdb":
            cls_num = 23

        # self.missing_type = missing_type
        # define freezing component
        self.embedding_layer = vilt.embeddings
        self.encoder_layer = vilt.encoder.layer
        self.layernorm = vilt.layernorm
        self.freeze()

        # missing_aware_prompt

        # complete prompt define
        complete_prompt = torch.zeros(prompt_num, prompt_len, embed_dim)
        complete_prompt[:, 0:1, :].fill_(1)
        if prompt_type == 'attention' and learnt_p:
            complete_prompt[:, prompt_len // 2:prompt_len // 2 + 1, :].fill_(1)
        complete_prompt = nn.Parameter(complete_prompt)

        # text missing prompt define
        missing_text_prompt = torch.zeros(prompt_num, prompt_len, embed_dim)
        missing_text_prompt[:, 2:3, :].fill_(1)
        if prompt_type == 'attention' and learnt_p:
            missing_text_prompt[:, prompt_len // 2 + 2:prompt_len // 2 + 3, :].fill_(1)
        missing_text_prompt = nn.Parameter(missing_text_prompt)

        # image missing prompt define
        missing_img_prompt = torch.zeros(prompt_num, prompt_len, embed_dim)
        missing_img_prompt[:, 1:2, :].fill_(1)
        if prompt_type == 'attention' and learnt_p:
            missing_img_prompt[:, prompt_len // 2 + 1:prompt_len // 2 + 2, :].fill_(1)
        missing_img_prompt = nn.Parameter(missing_img_prompt)

        # missing_aware_prompt define
        if missing_type == "text":
            self.missing_aware_prompt = torch.cat([missing_text_prompt.unsqueeze(0),complete_prompt.unsqueeze(0)],
                                                  dim=0)
        elif missing_type == "image":
            self.missing_aware_prompt = torch.cat([missing_img_prompt.unsqueeze(0),complete_prompt.unsqueeze(0)],
                                                  dim=0)
        elif missing_type == "both":
            self.missing_aware_prompt = torch.cat(
                [missing_text_prompt.unsqueeze(0),missing_img_prompt.unsqueeze(0),complete_prompt.unsqueeze(0)],
                dim=0)
        else:
            raise ValueError("missing_type should be 'Text', 'Image' or 'Both'")
        
        self.missing_aware_prompt = self.missing_aware_prompt.to(device)

        if not learnt_p:
            self.missing_aware_prompt.requires_grad = False

        # define training component
        self.pooler = vilt.pooler
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, cls_num),
        )
        self.classifier.apply(init_weights)

    def freeze(self):
        for param in self.embedding_layer.parameters():
            param.requires_grad = False
        for param in self.encoder_layer.parameters():
            param.requires_grad = False
        for param in self.layernorm.parameters():
            param.requires_grad = False

    def forward(self,
                input_ids,
                pixel_values,
                pixel_mask,
                token_type_ids,
                attention_mask,
                missing_mask,
                image_token_type_idx=1):
        output, attention_mask = self.embedding_layer(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         token_type_ids=token_type_ids,
                                                         inputs_embeds=None,
                                                         image_embeds=None,
                                                         pixel_values=pixel_values,
                                                         pixel_mask=pixel_mask,
                                                         image_token_type_idx=image_token_type_idx)
        missing_aware_prompt = self.missing_aware_prompt[missing_mask]
        N = missing_aware_prompt.size(0)
        temp_mask = attention_mask
        
        if self.prompt_type == "input":
            for i, layer_module in enumerate(self.encoder_layer):
                if i <= self.prompt_num - 1:
                    missing_aware_prompt_mask = torch.ones(N, self.prompt_len).to(self.device)
                    temp_mask = torch.cat([missing_aware_prompt_mask, temp_mask], dim=1)
                    attention_mask = temp_mask.unsqueeze(1).unsqueeze(3)
                    layer_outputs = layer_module(torch.cat([missing_aware_prompt[:,i,:,:],output],dim=1),
                                                 attention_mask=attention_mask
                                                 )
                    output = layer_outputs[0]
                else:
                    layer_outputs = layer_module(output, attention_mask=attention_mask)
                    output = layer_outputs[0]
                    
        elif self.prompt_type == "attention":
            for i, layer_module in enumerate(self.encoder_layer):
                if i <= self.prompt_num - 1:
                    layer_outputs = layer_module(output,
                                                 learnt_p = self.learnt_p,
                                                 missing_aware_prompt = missing_aware_prompt[:,i,:,:],
                                                 prompt_type=self.prompt_type,
                                                 attention_mask=attention_mask.unsqueeze(1).unsqueeze(-1)
                                                 )
                    output = layer_outputs[0]
                else:
                    layer_outputs = layer_module(output,
                                                 attention_mask=attention_mask.unsqueeze(1).unsqueeze(-1)
                                                 )
                    output = layer_outputs[0]

        output = self.layernorm(output)
        if self.prompt_type == "input":
            output = self.pooler(output[:, self.total_prompt_len:self.total_prompt_len + 1])
        elif self.prompt_type == "attention":
            output = self.pooler(output)
        output = self.classifier(output)
        return {
            "output": output
        }


if __name__ == "__main__":
    text = ['This is a test sentence.', 'This is another test sentence.']
    tokenizer = BertTokenizer.from_pretrained(r'vilt/weights/mlm', do_lower_case=True)
    text_encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_special_tokens_mask=True,
    )
    input_ids = text_encoding['input_ids']
    attention_mask = text_encoding['attention_mask']
    token_type_ids = text_encoding['token_type_ids']
    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.int64)
    attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
    image = Image.open(r'D:\AIDP\archive\images\train\apple_pie\apple_pie_0.jpg')
    image_processor = ViltImageProcessor.from_pretrained(r'vilt/weights/mlm')
    image = [image, image]
    encoding_image_processor = image_processor(image, return_tensors="pt")
    pixel_values = encoding_image_processor["pixel_values"]
    pixel_mask = encoding_image_processor["pixel_mask"]
    input_ids = input_ids.to('cuda:0')
    token_type_ids = token_type_ids.to('cuda:0')
    attention_mask = attention_mask.to('cuda:0')
    pixel_values = pixel_values.to('cuda:0')
    pixel_mask = pixel_mask.to('cuda:0')
    map = maps(vilt=ViltModel.from_pretrained(r'vilt/weights/mlm'),
              dataset_name='mmimdb',
              cls_num=23,
              max_text_len=128,
              learnt_p=True,
              missing_type='Text',
              prompt_type='input',
              prompt_num=5,
              prompt_len=16,
              device='cuda:0')
    map = map.to('cuda:0')
    missing_mask = torch.tensor([0, 1], dtype=torch.int64).to('cuda:0')
    output = map(input_ids=input_ids,
                 pixel_values=pixel_values,
                 pixel_mask=pixel_mask,
                 token_type_ids=token_type_ids,
                 attention_mask=attention_mask,
                 missing_mask=missing_mask,
                 image_token_type_idx=1)
