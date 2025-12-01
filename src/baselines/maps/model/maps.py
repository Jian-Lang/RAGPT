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
        self.missing_type = missing_type
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
            missing_aware_prompt = torch.cat([missing_text_prompt.unsqueeze(0),complete_prompt.unsqueeze(0)],
                                                  dim=0)
        elif missing_type == "image":
            missing_aware_prompt = torch.cat([missing_img_prompt.unsqueeze(0),complete_prompt.unsqueeze(0)],
                                                  dim=0)
        elif missing_type == "both":
            missing_aware_prompt = torch.cat(
                [missing_text_prompt.unsqueeze(0),missing_img_prompt.unsqueeze(0),complete_prompt.unsqueeze(0)],
                dim=0)
        else:
            raise ValueError("missing_type should be 'Text', 'Image' or 'Both'")
        
        # Wrap as nn.Parameter to register in the model's para  meters
        self.missing_aware_prompt = nn.Parameter(missing_aware_prompt.to(device))

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

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype = None) -> torch.Tensor:
        """
        Makes broadcastable attention mask so that masked tokens are ignored.
        
        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            dtype (`torch.dtype`, *optional*):
                The dtype of the extended attention mask.
                
        Returns:
            `torch.Tensor` The extended attention mask, with 0.0 for positions to attend 
            and the dtype's smallest value for masked positions.
        """
        if dtype is None:
            dtype = torch.float32
            
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.shape})"
            )
        
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def apply_missing_mask_to_attention(self, attention_mask: torch.Tensor, missing_mask) -> torch.Tensor:
        """
        Modify attention_mask based on missing_mask for image missing cases.
        
        Arguments:
            attention_mask (`torch.Tensor`):
                Attention mask from embedding layer, shape (batch_size, text_len + image_len).
            missing_mask:
                Missing mask indicating which modality is missing for each sample.
                - For "both": 0=text missing, 1=image missing, 2=complete
                - For "image": 0=image missing, 1=complete
                
        Returns:
            `torch.Tensor` Modified attention mask with image parts masked for missing samples.
        """
        # Convert missing_mask to tensor if needed
        if not isinstance(missing_mask, torch.Tensor):
            missing_mask_tensor = torch.tensor(missing_mask, device=self.device)
        else:
            missing_mask_tensor = missing_mask
        
        # Only modify image part of attention_mask for image missing cases
        if self.missing_type == "both":
            # For image missing (missing_mask == 1), set image part to 0
            image_missing_indices = (missing_mask_tensor == 1).nonzero(as_tuple=True)[0]
            if len(image_missing_indices) > 0:
                attention_mask[image_missing_indices, self.max_text_len:] = 0
                
        elif self.missing_type == "image":
            # For image missing (missing_mask == 0), set image part to 0
            image_missing_indices = (missing_mask_tensor == 0).nonzero(as_tuple=True)[0]
            if len(image_missing_indices) > 0:
                attention_mask[image_missing_indices, self.max_text_len:] = 0
        
        return attention_mask

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
        
        # Apply missing mask to attention mask
        attention_mask = self.apply_missing_mask_to_attention(attention_mask, missing_mask)
        
        missing_aware_prompt = self.missing_aware_prompt[missing_mask]
        N = missing_aware_prompt.size(0)
        temp_mask = attention_mask
        dtype = output.dtype
        
        if self.prompt_type == "input":
            for i, layer_module in enumerate(self.encoder_layer):
                if i <= self.prompt_num - 1:
                    missing_aware_prompt_mask = torch.ones(N, self.prompt_len).to(self.device)
                    temp_mask = torch.cat([missing_aware_prompt_mask, temp_mask], dim=1)
                    extended_attention_mask = self.get_extended_attention_mask(temp_mask, dtype=dtype)
                    layer_outputs = layer_module(torch.cat([missing_aware_prompt[:,i,:,:],output],dim=1),
                                                 attention_mask=extended_attention_mask
                                                 )
                    output = layer_outputs[0]
                else:
                    extended_attention_mask = self.get_extended_attention_mask(temp_mask, dtype=dtype)
                    layer_outputs = layer_module(output, attention_mask=extended_attention_mask)
                    output = layer_outputs[0]
                    
        elif self.prompt_type == "attention":
            P = self.prompt_len // 2
            prompt_mask = torch.ones(N, P).to(self.device)
            attention_mask_with_prompt = torch.cat([prompt_mask, attention_mask], dim=1)
            extended_attention_mask = self.get_extended_attention_mask(attention_mask_with_prompt, dtype=dtype)
            
            for i, layer_module in enumerate(self.encoder_layer):
                if i <= self.prompt_num - 1:
                    layer_outputs = layer_module(output,
                                                 learnt_p = self.learnt_p,
                                                 missing_aware_prompt = missing_aware_prompt[:,i,:,:],
                                                 prompt_type=self.prompt_type,
                                                 attention_mask=extended_attention_mask
                                                 )
                    output = layer_outputs[0]
                else:
                    # After prompt layers, use original attention mask without prompt extension
                    extended_attention_mask_no_prompt = self.get_extended_attention_mask(attention_mask, dtype=dtype)
                    layer_outputs = layer_module(output,
                                                 attention_mask=extended_attention_mask_no_prompt
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