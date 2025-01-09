import torch
import torch.nn as nn
import torch.nn.functional as F
# abbreviation: MMG: Missing Modality Generator, CAP: Context-Aware Prompter

class MMG(nn.Module):
    def __init__(self, dropout_rate, n, d):
        super(MMG, self).__init__()
        self.n = n
        self.d = d
        self.W = nn.Parameter(torch.randn(n, d, dtype=torch.cfloat))  
        self.layer_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(d, d)

    def forward(self, F_l):
        F_l = torch.mean(F_l, dim=1)
        X_l = torch.fft.fft(F_l, dim=1)  
        X_tilde_l = self.W * X_l
        F_tilde_l = torch.fft.ifft(X_tilde_l, dim=1).real
        F_l = self.layer_norm(F_l + self.dropout(F_tilde_l))
        F_l = self.linear(F_l)
        return F_l


class CAP(nn.Module):
    def __init__(self, prompt_length,dim=768):
        super(CAP, self).__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.pooling = nn.AdaptiveAvgPool2d((prompt_length, dim))

    def forward(self, V, T, r_i, r_t):
        V_to_V = self.attention(V, r_i)
        T_to_T = self.attention(T, r_t)
        return T_to_T, V_to_V

    def attention(self, query, key_value):
        b, k, s, _ = key_value.shape

        q = self.q_proj(query).unsqueeze(1).expand(b, k, -1, -1)  
        k = self.k_proj(key_value)  
        v = self.v_proj(key_value) 

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)  
        attn_probs = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_probs, v)  
        output = self.pooling(output)
        output = output.mean(dim=1)
        return output