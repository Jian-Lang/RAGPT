import torch
from torch import nn

from baselines.msps.vilt import ViltModel


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class msps(torch.nn.Module):
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
                 **args):
        super(msps, self).__init__()
        self.missing_type = missing_type
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

        # modality specific prompt define

        # text prompt define
        text_prompt = torch.zeros(prompt_num, prompt_len, embed_dim)
        text_prompt[:, 2:3, :].fill_(1)
        if prompt_type == 'attention' and learnt_p:
            text_prompt[:, prompt_len // 2 + 2:prompt_len // 2 + 3, :].fill_(1)
        self.text_prompt = nn.Parameter(text_prompt)

        # image prompt define
        img_prompt = torch.zeros(prompt_num, prompt_len, embed_dim)
        img_prompt[:, 1:2, :].fill_(1)
        if prompt_type == 'attention' and learnt_p:
            img_prompt[:, prompt_len // 2 + 1:prompt_len // 2 + 2, :].fill_(1)
        self.img_prompt = nn.Parameter(img_prompt)

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
        text_prompt = self.text_prompt.unsqueeze(0)
        img_prompt = self.img_prompt.unsqueeze(0)
        if self.missing_type == "text":
            prompt = torch.cat([text_prompt,text_prompt + img_prompt], dim=0)
        elif self.missing_type == "image":
            prompt = torch.cat([img_prompt,text_prompt + img_prompt], dim=0)
        elif self.missing_type == "both":
            prompt = torch.cat([text_prompt,img_prompt,text_prompt + img_prompt], dim=0)
        missing_aware_prompt = prompt[missing_mask]
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
        text_prompt = torch.flatten(text_prompt)
        img_prompt = torch.flatten(img_prompt)

        numerator = torch.abs(torch.dot(text_prompt, img_prompt))
        epsilon=1e-8
        denominator = torch.max(torch.norm(text_prompt) * torch.norm(img_prompt), torch.tensor(epsilon))
        ortho_loss = numerator / denominator
        return {
            "output": output,
            "ortho_loss": ortho_loss
        }
