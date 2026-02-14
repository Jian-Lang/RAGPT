from transformers import BertTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('dandelin/vilt-b32-mlm', do_lower_case=True)

df = pd.read_pickle('dataset/food101/test.pkl')
texts = df['text'].tolist()

token_lengths = []
for t in tqdm(texts):
    enc = tokenizer(t, add_special_tokens=True, padding=False, truncation=False)
    token_lengths.append(len(enc['input_ids']))

# print("样本数:", len(token_lengths))
# print("最大 token 长度:", max(token_lengths))
# for q in [50, 75, 90, 95, 98, 99]:
#     print(f"{q}分位:", int(np.percentile(token_lengths, q)))

threshold = 158
num_exceeding = sum(1 for length in token_lengths if length > threshold)
print(f"超过 {threshold} token 的文本数量: {num_exceeding}")