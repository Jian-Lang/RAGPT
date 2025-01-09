# RAGPT
This repo is the official implementation of _Retrieval-Augmented Dynamic Prompt Tuning for Incomplete Multimodal Learning_ accepted by AAAI 2025. 

Arxiv link: https://arxiv.org/abs/2501.01120

## Abstract
Multimodal learning with incomplete modality is practical and challenging. Recently, researchers have focused on enhancing the robustness of pre-trained MultiModal Transformers (MMTs) under missing modality conditions by applying learnable prompts. However, these prompt-based methods face several limitations: (1) incomplete modalities provide restricted modal cues for task-specific inference, (2) dummy imputation for missing content causes information loss and introduces noise, and (3) static prompts are instance-agnostic, offering limited knowledge for instances with various missing conditions. To address these issues, we propose **RAGPT**, a novel **R**etrieval-**A**u**G**mented dynamic **P**rompt **T**uning framework. RAGPT comprises three modules: (I) the multi-channel retriever, which identifies similar instances through a withinmodality retrieval strategy, (II) the missing modality generator, which recovers missing information using retrieved contexts, and (III) the context-aware prompter, which captures contextual knowledge from relevant instances and generates dynamic prompts to largely enhance the MMT’s robustness. 

## Framework
<img width="1232" alt="image" src="https://github.com/user-attachments/assets/0a7e7510-076d-4dd0-99cd-dcec59dc775e" />

## Environment Configuration
First, clone this repo:
```shell
git clone https://github.com/Jian-Lang/RAGPT.git

cd RAGPT
```
First, create a new conda env for RAGPT:
```shell
conda create -n RAGPT python=3.9
```
Next, activate this env and install the dependencies from the requirements.txt:
```shell
conda activate RAGPT

pip install -r requirements.txt
```

## Data Preparation
### MM-IMDb
First, download the dataset from this link: https://archive.org/download/mmimdb/mmimdb.tar.gz

Then, place the raw images in folder **dataset/mmimdb/image** and put the json files in folder **dataset/mmimdb/meta_data**.
### HateMemes
First, download the dataset from this link: https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset

Then, place the raw images in folder **dataset/hatememes/image** and put the json files in folder **dataset/hatememes/metadata**.

Next, replace the **test.json** in metadata with **test_seen.json** downloaded from this link: https://www.kaggle.com/datasets/williamberrios/hateful-memes as the test.json downloaded from the prior website has no label information for evaluation. (Do not change other files, only replace the test.json with test_seen.json)
### Food101
First, download the dataset from this link: https://www.kaggle.com/datasets/gianmarco96/upmcfood101

Then, place the raw images in folder **dataset/mmimdb/image** and put the csv files in folder **dataset/mmimdb/meta_data**.

## Code Running
### Dataset Initiation
Run the following script to init the dataset:
```shell
sh src/scripts/init_data.sh
```

### Training & Evaluation
Run the following script to training the model and evaluate the results:
```shell
sh src/scripts/eval.sh
```
All the parameters have the same meaning as describe in our paper and you can simply config them in **src/config/config.yaml** or in command line.

If you find this repo help you, please give us a star ⭐⭐⭐.
