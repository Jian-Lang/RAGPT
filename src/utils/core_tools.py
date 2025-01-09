import os
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, AUROC
import torch
from torch.optim.lr_scheduler import LambdaLR
import logging
import os
import random
from datetime import datetime
import numpy as np
import torch
from model import RAGPT as Model
from model import ViltModel, ViltImageProcessor
import importlib
from PIL import Image
import pandas as pd
from transformers import BertTokenizer
import json
from tqdm import tqdm
from transformers import AutoModel,AutoProcessor
# abbreviation: MCR: Multi-Channel Retriever

def init_data_hatememes():
    for split in tqdm(['train', 'dev', 'test_seen']):
        data = pd.read_json(f'dataset/hatememes/meta_data/{split}.jsonl', lines=True)
        data.rename(columns={'id': 'item_id'}, inplace=True)
        data['item_id'] = data['item_id'].apply(lambda x: f"{int(x):05d}")
        if split == 'test_seen': split = 'test'
        if split == 'dev': split = 'valid'
        data.to_pickle(f'dataset/hatememes/{split}.pkl')

def init_data_mmimdb():
    data_split = json.load(open('dataset/mmimdb/split.json'))
    genres = ['Drama', 'Comedy', 'Romance', 'Thriller', 'Crime', 'Action', 'Adventure', 'Horror'
        , 'Documentary', 'Mystery', 'Sci-Fi', 'Fantasy', 'Family', 'Biography', 'War', 'History', 'Music',
        'Animation', 'Musical', 'Western', 'Sport', 'Short', 'Film-Noir'] 
    for split in tqdm(['train', 'dev', 'test']):
        item_id_list = data_split[split]
        item_id_list = [str(i) for i in item_id_list]
        text_list = []
        label_list = []
        for item_id in item_id_list:
            json_file = json.load(open(f'dataset/mmimdb/meta_data/{item_id}.json'))
            try:
                text = json_file['plot outline']
            except:
                text = json_file['plot'][0]
            text_list.append(text)
            label =  [1 if g in json_file['genres'] else 0 for g in genres]
            label_list.append(label)
            
        data = pd.DataFrame({'item_id': item_id_list, 'text': text_list, 'label': label_list})
        data.to_pickle(f'dataset/mmimdb/{split if split != "dev" else "valid"}.pkl')

def init_data_food101():
    class_idx = json.load(open('dataset/food101/class_idx.json'))

    for split in tqdm(['train', 'test']):
        data = pd.read_csv(f'dataset/food101/meta_data/{split}_titles.csv')
        data.columns = ['item_id', 'text', 'label']
        data['item_id'] = data['item_id'].apply(lambda x: x.split('.')[0])
        data['label'] = data['label'].apply(lambda x: class_idx[x])
        data.to_pickle(f'dataset/food101/{split}.pkl')

class MemoryBankGenerator(torch.nn.Module):
    def __init__(self, cfg):
        super(MemoryBankGenerator, self).__init__()
        pretrained_vilt = ViltModel.from_pretrained('dandelin/vilt-b32-mlm')
        self.embedding_layer = pretrained_vilt.embeddings
        self._freeze()
        self.tokenizer = BertTokenizer.from_pretrained('dandelin/vilt-b32-mlm', do_lower_case=True)
        self.image_processor = ViltImageProcessor.from_pretrained('dandelin/vilt-b32-mlm')
        self.dataset = cfg.data_para.dataset
        self.max_text_len = cfg.data_para.max_text_len
        self.max_image_len = cfg.data_para.max_image_len
        self.df_train = pd.read_pickle(rf'dataset/{self.dataset}/train.pkl')
        self.df_test = pd.read_pickle(rf'dataset/{self.dataset}/test.pkl')
        if self.dataset != "food101":
            self.df_valid = pd.read_pickle(rf'dataset/{self.dataset}/valid.pkl')
        self.batch_size = cfg.batch_size
        # check if memory_bank folder exists
        if not os.path.exists(f'dataset/memory_bank/{self.dataset}/text'):
            os.makedirs(f'dataset/memory_bank/{self.dataset}/text')
        if not os.path.exists(f'dataset/memory_bank/{self.dataset}/image'):
            os.makedirs(f'dataset/memory_bank/{self.dataset}/image')

    def _freeze(self):
        for param in self.embedding_layer.parameters():
            param.requires_grad = False

    def _encode(self, input_ids, pixel_values, pixel_mask, token_type_ids, attention_mask, image_token_type_idx=1):
        embedding, attention_mask = self.embedding_layer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            image_embeds=None,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            image_token_type_idx=image_token_type_idx
        )
        return embedding

    def _resize_image(self, img, size=(384, 384)):
        return img.resize(size, Image.BILINEAR)

    def _process_batch(self, df, start_idx, end_idx):
        texts = df['text'][start_idx:end_idx]
        ids = df['item_id'][start_idx:end_idx]

        text_encodings = self.tokenizer(
            texts.tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        input_ids = text_encodings['input_ids']
        attention_mask = text_encodings['attention_mask']
        token_type_ids = text_encodings['token_type_ids']

        images = []
        for id in ids:
            if self.dataset == "hatememes":
                image_path = fr'dataset/{self.dataset}/image/{id}.png'
            elif self.dataset == "food101":
                image_path = fr'dataset/{self.dataset}/image/{id}.jpg'
            elif self.dataset == "mmimdb":
                image_path = fr'dataset/{self.dataset}/image/{id}.jpeg'
            image = Image.open(image_path).convert("RGB")
            image = self._resize_image(image)
            images.append(image)
        
        encoding_image_processor = self.image_processor(images, return_tensors="pt")
        pixel_values = encoding_image_processor["pixel_values"]
        pixel_mask = encoding_image_processor["pixel_mask"]

        emb = self._encode(input_ids, pixel_values, pixel_mask, token_type_ids, attention_mask)
        text_emb = emb[:, :self.max_text_len]
        image_emb = emb[:, self.max_text_len:]

        for i, id in enumerate(ids):
            np.save(f'dataset/memory_bank/{self.dataset}/text/{id}.npy', text_emb[i].detach().numpy())
            np.save(f'dataset/memory_bank/{self.dataset}/image/{id}.npy', image_emb[i].detach().numpy())
    
    def run(self):
        for i in tqdm(range(0, len(self.df_train), self.batch_size)):
            start_idx = i
            end_idx = min(i + self.batch_size, len(self.df_train))
            self._process_batch(self.df_train, start_idx, end_idx)
        
        if self.dataset != "food101":
            for i in tqdm(range(0, len(self.df_valid), self.batch_size)):
                start_idx = i
                end_idx = min(i + self.batch_size, len(self.df_valid))
                self._process_batch(self.df_valid, start_idx, end_idx)

class MCR():
    def __init__(self, cfg):
        self.dataset = cfg.data_para.dataset
        self.batch_size = cfg.batch_size
        self.top_k = 20
        self.img_path = os.path.join('dataset', self.dataset, 'image')
        self.img_name_list = os.listdir(self.img_path)
        self.device = cfg.device
        self.pretrained_model = AutoModel.from_pretrained('openai/clip-vit-large-patch14-336')
        self.processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
        self.pretrained_model = self.pretrained_model.to(self.device)
        
        self.df_train = pd.read_pickle(os.path.join('dataset', self.dataset, 'train.pkl'))
        self.df_test = pd.read_pickle(os.path.join('dataset', self.dataset, 'test.pkl'))
        if self.dataset != 'food101':
            self.df_valid = pd.read_pickle(os.path.join('dataset', self.dataset, 'valid.pkl'))

    def _compute_similarity_in_batches(self, query_vectors, memory_bank, memory_bank_id, memory_bank_label):
        r_id_list = []
        sims_list = []
        r_label_list = []
        for i in tqdm(range(0, len(query_vectors), self.batch_size)):
            batch = query_vectors[i:i+self.batch_size].unsqueeze(1)
            similarity = F.cosine_similarity(batch, memory_bank.unsqueeze(0), dim=-1)
            sim_scores, top_k_id = torch.topk(similarity, k=self.top_k, dim=-1)
            for j in range(batch.size(0)):
                id_index = i + j
                id = memory_bank_id[id_index] if id_index < len(memory_bank_id) else None
                retrieved_ids = [memory_bank_id[idx] for idx in top_k_id[j].tolist() if memory_bank_id[idx] != id]
                retrieved_labels = [memory_bank_label[idx] for idx in top_k_id[j].tolist() if memory_bank_id[idx] != id]
                sim_score = sim_scores[j,1:]
                if len(retrieved_ids) > self.top_k:
                    retrieved_ids = retrieved_ids[:self.top_k]
                    sim_score = sim_score[:self.top_k]
                    retrieved_labels = retrieved_labels[:self.top_k]
                r_id_list.append(retrieved_ids)
                sims_list.append(sim_score.tolist())
                r_label_list.append(retrieved_labels)
        return  r_id_list,sims_list, r_label_list

    def _encode_text(self, text):
        inputs = self.processor(text=text, return_tensors="pt", padding=True,truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        text_features = self.pretrained_model.text_model.embeddings(input_ids, attention_mask)
        text_features = self.pretrained_model.text_model.encoder(text_features).last_hidden_state
        text_features = self.pretrained_model.text_model.final_layer_norm(text_features)
        text_features = (self.pretrained_model.text_projection(text_features))
        return text_features[0, -1, :]

    def _encode_image(self, images):
        with torch.no_grad():
            processed_images = self.processor(images=images, return_tensors="pt").to(self.device)
            image_features = self.pretrained_model.vision_model.embeddings(processed_images['pixel_values'])
            image_features = self.pretrained_model.vision_model.pre_layrnorm(image_features)
            image_features = self.pretrained_model.vision_model.encoder(image_features).last_hidden_state
            image_features = self.pretrained_model.vision_model.post_layernorm(image_features)
            image_features = self.pretrained_model.visual_projection(image_features)
            return image_features[:, 0, :]
    
    def _retrieval_vector_generation(self):
        img_name_list = os.listdir(self.img_path)
        print("Loading all images...")
        all_images = [Image.open(os.path.join(self.img_path, img_name)) for img_name in tqdm(img_name_list, desc="Loading Images")]
        print("==> All images loaded successfully!")

        images = []
        batch_size = self.batch_size
        for i in tqdm(range(0, len(all_images), batch_size), desc="Encoding Image Batches"):
            batch_images = all_images[i:i+batch_size]
            batch_outputs = self._encode_image(batch_images)
            images.extend(batch_outputs.cpu().tolist())
        print("==> Image encoding done!")
        if self.dataset != "mmimdb":
            id_list = [str(x[:-4]) for x in img_name_list]
        else:
            id_list = [str(x[:-5]) for x in img_name_list]
        df_img_query = pd.DataFrame({'item_id': id_list, 'q_i': images})
        self.df_train = pd.merge(self.df_train, df_img_query, on='item_id', how='inner')
        self.df_test = pd.merge(self.df_test, df_img_query, on='item_id', how='inner')
        if self.dataset != 'food101':
            self.df_valid = pd.merge(self.df_valid, df_img_query, on='item_id', how='inner')

        q_t_list = [self._encode_text(text).tolist() for text in tqdm(self.df_train['text'], desc=f"Encoding Text")]
        self.df_train['q_t'] = q_t_list
        q_t_list = [self._encode_text(text).tolist() for text in tqdm(self.df_test['text'], desc=f"Encoding Text")]
        self.df_test['q_t'] = q_t_list

        if self.dataset != 'food101':
            q_t_list = [self._encode_text(text).tolist() for text in tqdm(self.df_valid['text'], desc=f"Encoding Text")]
            self.df_valid['q_t'] = q_t_list
        print("==> Text encoding done!")

    def _within_retrieval(self):
        train_q_i = self.df_train[f'q_i'].tolist()
        train_q_t = self.df_train[f'q_t'].tolist()
        train_item_id = self.df_train['item_id'].tolist()
        train_label = self.df_train['label'].tolist()

        if self.dataset != 'food101':
            valid_q_i = self.df_valid[f'q_i'].tolist()
            valid_q_t = self.df_valid[f'q_t'].tolist()
            valid_item_id = self.df_valid['item_id'].tolist()
            valid_label = self.df_valid['label'].tolist()

        test_q_i = self.df_test[f'q_i'].tolist()
        test_q_t = self.df_test[f'q_t'].tolist()

        if self.dataset != 'food101':
            r_v_i = train_q_i + valid_q_i
            r_v_t = train_q_t + valid_q_t
            memory_bank_id = train_item_id + valid_item_id
            memory_bank_label = train_label + valid_label

        else:
            r_v_i = train_q_i 
            r_v_t = train_q_t 
            memory_bank_id = train_item_id 
            memory_bank_label = train_label

        r_v_i = torch.tensor(r_v_i).squeeze(1).to(self.device)
        r_v_t = torch.tensor(r_v_t).squeeze(1).to(self.device)
        test_q_i = torch.tensor(test_q_i).squeeze(1).to(self.device)
        test_q_t = torch.tensor(test_q_t).squeeze(1).to(self.device)
        train_q_i = torch.tensor(train_q_i).squeeze(1).to(self.device)
        train_q_t = torch.tensor(train_q_t).squeeze(1).to(self.device)

        self.df_train[f't2t_id_list'], self.df_train[f't2t_sims_list'], self.df_train['t2t_label_list'] = self._compute_similarity_in_batches(train_q_t,r_v_t, memory_bank_id, memory_bank_label)
        self.df_train[f'i2i_id_list'], self.df_train[f'i2i_sims_list'], self.df_train['i2i_label_list'] = self._compute_similarity_in_batches(train_q_i,r_v_i, memory_bank_id, memory_bank_label)
        self.df_test[f't2t_id_list'], self.df_test[f't2t_sims_list'], self.df_test['t2t_label_list'] = self._compute_similarity_in_batches(test_q_t,r_v_t,memory_bank_id, memory_bank_label)
        self.df_test[f'i2i_id_list'], self.df_test[f'i2i_sims_list'], self.df_test['i2i_label_list'] = self._compute_similarity_in_batches(test_q_i,r_v_i,memory_bank_id, memory_bank_label)
        if self.dataset != 'food101':
            valid_q_i = torch.tensor(valid_q_i).squeeze(1).to(self.device)
            valid_q_t = torch.tensor(valid_q_t).squeeze(1).to(self.device)
            self.df_valid[f't2t_id_list'], self.df_valid[f't2t_sims_list'], self.df_valid['t2t_label_list'] = self._compute_similarity_in_batches(valid_q_t, r_v_t, memory_bank_id, memory_bank_label)
            self.df_valid[f'i2i_id_list'], self.df_valid[f'i2i_sims_list'], self.df_valid['i2i_label_list'] = self._compute_similarity_in_batches(valid_q_i, r_v_i, memory_bank_id, memory_bank_label)
            self.df_valid.to_pickle(os.path.join(os.path.join('dataset', self.dataset, 'valid.pkl')))

        self.df_train.to_pickle(os.path.join(os.path.join('dataset', self.dataset, 'train.pkl')))
        self.df_test.to_pickle(os.path.join(os.path.join('dataset', self.dataset, 'test.pkl')))
        print(f"==> Saved retrieval results for {self.dataset}!")

    def run(self):
        self._retrieval_vector_generation()
        self._within_retrieval()

def generate_missing_table(missing_rate, missing_type, dataset, base_file_path='dataset/missing_table', **kargs):
    
    assert missing_type in ['Text', 'Image', 'Both'], "Invalid missing type"
    assert 0 <= missing_rate <= 1, "Invalid missing rate"
    if missing_type == 'Text' or missing_type == 'Image':
        missing_type = 'single'
    else:
        missing_type = 'both'

    file_path = f"{base_file_path}/{missing_type}/{dataset}/missing_table.pkl"
    folder = os.path.dirname(file_path)
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    if os.path.exists(file_path):
        df = pd.read_pickle(file_path)
        if f"missing_mask_{int(missing_rate* 10)}" in df.columns:
            df.drop(f"missing_mask_{int(missing_rate* 10)}", axis=1, inplace=True)
        print("File already exists, regenerating new missing column...")
    else:
        if dataset != 'food101':
            df = pd.concat([pd.read_pickle(f'dataset/{dataset}/{split}.pkl') for split in ['train', 'valid', 'test']])
        else:
            df = pd.concat([pd.read_pickle(f'dataset/{dataset}/{split}.pkl') for split in ['train', 'test']])
        df = df[['item_id']]
        print("File does not exist, generating new missing table and column...")

    if missing_type == 'single':
        num_missing = int(len(df) * missing_rate)  
        missing_mask = np.ones(len(df), dtype=int)  # Initialize as all 1s  
        missing_indices = np.random.choice(len(df), num_missing, replace=False)  # Randomly select missing indices
        
        missing_mask[missing_indices] = 0
    
    elif missing_type == 'both':
        num_missing = int(len(df) * missing_rate)  
        num_text_missing = num_missing // 2  
        num_visual_missing = num_missing // 2 
        
        missing_mask = np.full(len(df), 2, dtype=int)

        text_missing_indices = np.random.choice(len(df), num_text_missing, replace=False)
        missing_mask[text_missing_indices] = 0  
        
        remaining_indices = list(set(range(len(df))) - set(text_missing_indices))
        visual_missing_indices = np.random.choice(remaining_indices, num_visual_missing, replace=False)
        missing_mask[visual_missing_indices] = 1  
        
    df[f"missing_mask_{int(missing_rate* 10)}"] = missing_mask

    df.to_pickle(file_path)
    print(f"Missing table has been saved to {file_path}")

def resize_image(img, size=(384, 384)):
    return img.resize(size, Image.BILINEAR)

def load_model(**kargs):
    pretrained_vlit = ViltModel.from_pretrained('dandelin/vilt-b32-mlm')
    model = Model(vilt=pretrained_vlit, **kargs)
    return model

def get_dataset(dataset_name: str, **kargs):
    module = importlib.import_module(f"dataloader.{dataset_name}_dataset")
    if dataset_name == "hatememes":
        dataset_class = getattr(module, 'HatememesDataset')
    elif dataset_name == "mmimdb":
        dataset_class = getattr(module, 'MMIMDbDataset')
    elif dataset_name == "food101":
        dataset_class = getattr(module, 'Food101Dataset')
    dataset = dataset_class(**kargs)
    return dataset

def get_collator(max_text_len, **kargs):
    collator = Collator(max_text_len, **kargs)
    return collator

def get_evaluator(task_id, device):
    if task_id == "mmimdb":
        evaluator = MMIMDbMetric(device)
    elif task_id == "hatememes":
        evaluator = HatememesMetric(device)
    elif task_id == "food101":
        evaluator = Food101Metric(device)
    
    return evaluator

class Collator:
    def __init__(self, max_text_len, **kargs):
        self.image_processor = ViltImageProcessor.from_pretrained('dandelin/vilt-b32-mlm')
        self.tokenizer = BertTokenizer.from_pretrained('dandelin/vilt-b32-mlm', do_lower_case=True)
        self.max_text_len = max_text_len

    def __call__(self, batch):
        text = [item['text'] for item in batch]
        image = [item['image'] for item in batch]
        label = [item['label'] for item in batch]
        r_t_list = [item['r_t_list'] for item in batch]
        r_i_list = [item['r_i_list'] for item in batch]
        missing_mask = [item['missing_mask'] for item in batch]
        r_l_list = [item['r_l_list'] for item in batch]
        text_encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        input_ids = text_encoding['input_ids']
        attention_mask = text_encoding['attention_mask']
        token_type_ids = text_encoding['token_type_ids']
        image = [resize_image(img) for img in image]
        image_encoding = self.image_processor(image, return_tensors="pt")
        pixel_values = image_encoding["pixel_values"]
        pixel_mask = image_encoding["pixel_mask"]
        input_ids = torch.tensor(input_ids,dtype=torch.int64)
        token_type_ids = torch.tensor(token_type_ids,dtype=torch.int64)
        attention_mask = torch.tensor(attention_mask,dtype=torch.int64)
        label = torch.tensor(label,dtype=torch.float)
        r_l_list = torch.tensor(r_l_list,dtype=torch.long)
        r_t_list = torch.tensor(r_t_list,dtype=torch.float)
        r_i_list = torch.tensor(r_i_list,dtype=torch.float)
        return {
            "input_ids": torch.tensor(input_ids,dtype=torch.int64),
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "label": label,
            "r_t_list": r_t_list,
            "r_i_list": r_i_list,
            "missing_mask": torch.tensor(missing_mask,dtype=torch.int64),
            "r_l_list": r_l_list
        }

class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

def seed_init(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def print_init_msg(logger, cfg):
    logger.info('Random Seed: ' + f"{cfg.seed} ")
    logger.info('Device: ' + f"{cfg.device} ")
    logger.info('Model: ' + f"{cfg.model_para.model} ")
    logger.info('Backbone: ' + f"{cfg.model_para.backbone}")
    logger.info("Dataset: " + f"{cfg.data_para.dataset}")
    logger.info("Optimizer: " + f"{cfg.optim_para.name}(lr = {cfg.optim_para.lr})")
    logger.info("Weight Decay: " + f"{cfg.optim_para.weight_decay}")
    logger.info("Use Warmup: " + f"{cfg.optim_para.use_warmup}")
    logger.info("Warmup Rate: " + f"{int(cfg.optim_para.warmup_rate * 100)}%")
    logger.info("Total Epoch: " + f"{cfg.epochs} Turns")
    logger.info("Early Stop: " + f"{cfg.patience} Turns")
    logger.info("Batch Size: " + f"{cfg.batch_size}")
    logger.info("Number of Workers: " + f"{cfg.num_workers}")
    logger.info("Missing Rate: " + f"{cfg.data_para.missing_rate}")
    logger.info("Missing Type: " + f"{cfg.data_para.missing_type}")
    logger.info("K: " + f"{cfg.data_para.k}")
    logger.info("Prompt Length: " + f"{cfg.model_para.prompt_length}")
    logger.info("Prompt Position: " + f"{cfg.model_para.prompt_position}")
    logger.info("Training Starts!")

def make_saving_folder_and_logger(cfg):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"train_{cfg.model_para.model}_{cfg.data_para.dataset}_{cfg.data_para.missing_type}_{cfg.data_para.missing_rate}_{timestamp}"
    father_folder_name = cfg.save_path

    if not os.path.exists(father_folder_name):
        os.makedirs(father_folder_name)

    folder_path = os.path.join(father_folder_name, folder_name)
    os.mkdir(folder_path)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'{father_folder_name}/{folder_name}/log.txt')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return father_folder_name, folder_name, logger

def get_optim(max_steps, model, lr, weight_decay, warmup_rate=0.1, use_warmup=False, **kwargs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Compute warmup steps only if warmup is enabled
    warmup_steps = int(warmup_rate * max_steps) if use_warmup else 0

    def lr_lambda(current_step: int):
        if use_warmup and current_step < warmup_steps:
            # Linear warmup phase
            return float(current_step) / float(max(1, warmup_steps))
        # Linear decay phase
        return max(
            0.0,
            float(max_steps - current_step) / float(max(1, max_steps - warmup_steps)),
        )

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def compute_loss(output, label, task_id):
    if task_id == "mmimdb":
        loss = F.binary_cross_entropy_with_logits(output, label)
        return loss
    elif task_id == "hatememes":
        label = label.long()
        loss = F.cross_entropy(output, label)
        return loss
    elif task_id == "food101":
        label = label.long()
        loss = F.cross_entropy(output, label)
        return loss


class HatememesMetric:
    def __init__(self, device):
        self.auroc = AUROC(task="binary", num_classes=2).to(device)

    def _reset(self):
        self.auroc.reset()

    def update(self, preds, labels):
        labels = labels.long()
        self.auroc.update(preds, labels)

    def compute(self):
        auroc = self.auroc.compute()
        self._reset()
        return {"auroc": auroc}


class MMIMDbMetric:
    def __init__(self, device):
        self.f1_micro = F1Score(task="multiclass", average="micro", num_classes=23).to(device)
        self.f1_sample = F1Score(task="multiclass", average="samples", num_classes=23).to(device)

    def _reset(self):
        self.f1_micro.reset()
        self.f1_sample.reset()

    def update(self, preds, labels):
        labels = labels.long()
        self.f1_micro.update(preds, labels)
        self.f1_sample.update(preds, labels)

    def compute(self):
        f1_micro = self.f1_micro.compute()
        f1_sample = self.f1_sample.compute()
        self._reset()
        return {
            "f1_micro": f1_micro,
            "f1_sample": f1_sample,
        }


class Food101Metric:
    def __init__(self, device):
        self.accuracy = Accuracy(task="multiclass", num_classes=101).to(device)

    def _reset(self):
        self.accuracy.reset()

    def update(self, preds, labels):
        labels = labels.long()
        self.accuracy.update(preds, labels)

    def compute(self):
        accuracy = self.accuracy.compute()
        self._reset()
        return {
            "accuracy": accuracy,
        }
