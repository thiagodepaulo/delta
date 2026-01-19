import yaml
from delta.configs.dataset import DatasetConfig
from delta.data.dataset import PreferenceDataset
import os
import numpy as np
import json
import pandas as pd

def load_dataset_config(dataset_config_file, dts_name):
    with open(dataset_config_file, "r") as f:
        dc_ = yaml.safe_load(f)
        dataset_cfg = DatasetConfig(**dc_[dts_name])
    return dataset_cfg

def load_dataset(dataset_config: DatasetConfig, splits = ['train', 'dev', 'test', 'test_unseen']):
    results = {}
    for split in splits:        
        split_cfg = getattr(dataset_config, split)
        if split_cfg is not None:
            emb_file_name = os.path.join(dataset_config.dts_path or '', split_cfg.emb_file)
            texts_file_name = os.path.join(dataset_config.dts_path or '', split_cfg.text_file)
            df_file_name = os.path.join(dataset_config.dts_path or '', split_cfg.df_file)
            results[split] = load(emb_file_name, texts_file_name, df_file_name)
    return results    

def load(emb_file_name: str, texts_file_name: str, df_file_name: str):
    emb = np.load(emb_file_name)
    texts = json.load(open(texts_file_name))
    if df_file_name.endswith('.parquet'):
        df = pd.read_parquet(df_file_name)
    elif df_file_name.endswith('.jsonl'):
        print(df_file_name)
        df = pd.read_json(df_file_name, lines=True)    
    return {"embeddings": emb, "texts": texts, "df": df}

def create_torch_dataset(dataset_config_file, dts_name, splits = ['train', 'dev', 'test', 'test_unseen']):
    dts_cfg = load_dataset_config(dataset_config_file, dts_name)
    dts_raw = load_dataset(dts_cfg, splits)
    dts = {}
    for split in splits:
        if split in dts_raw.keys():
            print(f"Creating dataset for split: {split}")            
            dts[split] = PreferenceDataset(dts_raw[split]['df'], dts_raw[split]['embeddings'])
            print(f"Dataset {split} size: {len(dts[split])}")
    return dts
    