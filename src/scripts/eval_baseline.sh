python src/main.py --config-name=maps_config optim_para.weight_decay=5e-5 data_para.dataset_name='hatememes' data_para.missing_type='both' data_para.missing_rate=0.7 device='cuda:0'
python src/main.py --config-name=maps_config optim_para.weight_decay=5e-5 data_para.dataset_name='hatememes' data_para.missing_type='text' data_para.missing_rate=0.7 model_para device='cuda:0'
