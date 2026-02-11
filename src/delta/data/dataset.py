from delta.configs.dataset import DatasetConfig
from torch.utils.data import Dataset
import yaml
import os
import numpy as np
import pandas as pd
import torch
import json
from scipy import sparse


SPLITS = ['train', 'dev', 'test', 'test_unseen']
DEFAULT_REQUIRED_COLUMNS = ['u_id', 'prompt', 'chosen','rejected']
DEFAULT_EMB_COLUMNS_IDX = {
    'prompt_emb':'prompt_emb_idx', 
    'chosen_emb':'chosen_emb_idx', 
    'rejected_emb':'rejected_emb_idx'
    }

DEFAULT_BOW_COLUMNS_IDX = {
    'prompt_bow':'prompt_emb_idx', 
    'chosen_bow':'chosen_emb_idx', 
    'rejected_bow':'rejected_emb_idx'
    }

class PreferenceDataset(Dataset):
    
    def __init__(self, 
                 df, 
                 embeddings,
                 bow_embeddings=None,
                 required_columns = DEFAULT_REQUIRED_COLUMNS, 
                 emb_columns_idx= DEFAULT_EMB_COLUMNS_IDX, 
                 bow_columns_idx= DEFAULT_BOW_COLUMNS_IDX,
                 feature_columns=[], 
                 prefix_columns=[],
                 split="train"
                ):        
        self.df = df                        
        self.split = split
        
        self.bow_embeddings = bow_embeddings
        self.is_bow_sparse = sparse.issparse(bow_embeddings) if bow_embeddings is not None else False

        self.bow_columns_idx = bow_columns_idx if bow_embeddings is not None else None
        
        assert all(col in self.df.columns for col in required_columns), "DataFrame is missing required columns"
        if feature_columns:
            assert all(col in self.df.columns for col in feature_columns), "DataFrame is missing feature columns"
        if prefix_columns:
            prefix_columns = [col for col in self.df.columns if any(col.startswith(prefix) for prefix in prefix_columns)]
            assert all(col in self.df.columns for col in prefix_columns), "DataFrame is missing prefix columns"
        if emb_columns_idx:
            assert all(col in self.df.columns for col in emb_columns_idx.values()), "DataFrame is missing embedding index columns"
        if bow_columns_idx and bow_embeddings is not None:
            assert all(col in self.df.columns for col in bow_columns_idx.values()), "DataFrame is missing bow embedding index columns"
        
        self.features = None
        if feature_columns or prefix_columns:
            feature_columns = feature_columns + prefix_columns            
            self.features = self.df[feature_columns].to_numpy().astype(np.float32)
            self.features = torch.tensor(self.features, dtype=torch.float)
            self.n_features = self.features.shape[1]
            self.feature_columns = feature_columns
            print(f"Using {self.n_features} features columns: {self.feature_columns}")
            
                
        self.embeddings_map = {}        
        for k, idxs in emb_columns_idx.items():
            self.embeddings_map[k] = torch.tensor(embeddings[self.df[idxs]], dtype=torch.float)    
            
        self.bow_embeddings_map = {}        
        if bow_columns_idx and bow_embeddings is not None:
            for k, idxs in bow_columns_idx.items():
                # Store indices for sparse, or convert to dense for small data
                self.bow_embeddings_map[k] = self.df[idxs].values
        
                    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        item = {
            'u_id': torch.tensor(row['u_id'], dtype=torch.long),
            'prompt': row['prompt'],
            'chosen': row['chosen'],
            'rejected': row['rejected']
        }                
        
        if self.features is not None:
            item['features'] = self.features[idx]
        
        for k in self.embeddings_map:
            item[k] = self.embeddings_map[k][idx]        
        
        if self.bow_embeddings_map:
            for k in self.bow_embeddings_map:
                bow_idx = self.bow_embeddings_map[k][idx]
                if self.is_bow_sparse:
                    # Convert sparse row to dense tensor
                    bow_vector = self.bow_embeddings[bow_idx].toarray().squeeze()
                else:
                    bow_vector = self.bow_embeddings[bow_idx]
                item[k] = torch.tensor(bow_vector, dtype=torch.float)
                
        ## create y labels for preferred (chosen) vs non-preferred (rejected)        
        item['chosen_label'] = torch.tensor([1.0], dtype=torch.float) if self.split == "train" else torch.zeros(1)  # Preferred (chosen) is labeled as 1
        item['rejected_label'] = torch.tensor([0.0], dtype=torch.float) if self.split == "train" else torch.zeros(1)  # Non-preferred (rejected) is labeled as 0
        
        return item
    
def load_dataset_config(dataset_config_file, dts_name):
    with open(dataset_config_file, "r") as f:
        dc_ = yaml.safe_load(f)
        dataset_cfg = DatasetConfig(**dc_[dts_name])
    return dataset_cfg

def load_dataset(dataset_config: DatasetConfig, splits = SPLITS, has_bow=False):
    results = {}
    for split in splits:        
        split_cfg = getattr(dataset_config, split)
        if split_cfg is not None:
            emb_file_name = os.path.join(dataset_config.dts_path or '', split_cfg.emb_file)
            texts_file_name = os.path.join(dataset_config.dts_path or '', split_cfg.text_file)
            df_file_name = os.path.join(dataset_config.dts_path or '', split_cfg.df_file)
            bow_file_name = os.path.join(dataset_config.dts_path or '', split_cfg.bow_file) if has_bow else None
            results[split] = load(emb_file_name, texts_file_name, df_file_name, bow_file_name)
    return results    

def load(emb_file_name: str, texts_file_name: str, df_file_name: str, bow_file_name: str = None):
    emb = np.load(emb_file_name)
    bow_path = None
    print(f"Loading BOW embeddings from {bow_file_name}")
    # Check if it's sparse or dense
    if bow_file_name.endswith('.npz'):
        bow = sparse.load_npz(bow_file_name)  # Load sparse
        print(f"Loaded sparse BoW with shape {bow.shape}")
    else:
        bow = np.load(bow_file_name)  # Load dense
        print(f"Loaded dense BoW with shape {bow.shape}")
    texts = json.load(open(texts_file_name))
    if df_file_name.endswith('.parquet'):
        df = pd.read_parquet(df_file_name)
    elif df_file_name.endswith('.jsonl'):
        print(df_file_name)
        df = pd.read_json(df_file_name, lines=True)    
    return {"embeddings": emb, "texts": texts, "df": df, "bow": bow}

def create_torch_dataset(args, splits = SPLITS):
    dataset_config_file = args.dts_config_file 
    dts_name = args.dts_name    
    dts_cfg = load_dataset_config(dataset_config_file, dts_name)
    has_bow = getattr(args, 'has_bow', False)
    dts_raw = load_dataset(dts_cfg, splits, has_bow)
    dts = {}
    for split in splits:        
        if split in dts_raw.keys():
            print(f"Creating dataset for split: {split}")            
            dts[split] = PreferenceDataset(dts_raw[split]['df'], 
                                           dts_raw[split]['embeddings'], 
                                           bow_embeddings=dts_raw[split]['bow'],
                                           prefix_columns=dts_cfg.prefix_columns,
                                           split=split
                                           )
            print(f"Dataset {split} size: {len(dts[split])}")
    return dts

def load_vocab(dataset_config_file, dts_name):
    dts_cfg = load_dataset_config(dataset_config_file, dts_name)
    vocab_file = os.path.join(dts_cfg.dts_path or '', dts_cfg.vocab_file)
    vocab = []
    with open(vocab_file, 'r') as f:
        for line in f:
            vocab.append(line.strip())
    return vocab

