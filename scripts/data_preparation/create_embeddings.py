from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
import os
import sys
from optparse import OptionParser
import numpy as np
from openai import OpenAI
import json
import yaml
from delta.data.config import DatasetConfig


def main(args):
    usage = "%prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("--dts_name", type=str, help="Dataset name", default="prism")    
    parser.add_option("--dts_file_config", type=str, help="Dataset config file", default='dataset_configs.yaml')
    
    (options, args) = parser.parse_args(args)    
    
    with open(options.dts_file_config, "r") as f:
        dc_ = yaml.safe_load(f)
        dataset_cfg = DatasetConfig(**dc_[options.dts_name])
     
    print(dataset_cfg)   
    df_dict = load_dataframe(dataset_cfg)
    
    for split_name, df in df_dict.items():
        print(f"Creating embeddings for {split_name} split with {len(df)} rows.")
        split_cfg = getattr(dataset_cfg, split_name)
        
        emb_file_name = os.path.join(dataset_cfg.dts_path or '', split_cfg.emb_file)
        texts_file_name = os.path.join(dataset_cfg.dts_path or '', split_cfg.text_file)
        df_file_name = os.path.join(dataset_cfg.dts_path or '', split_cfg.df_file.replace('.jsonl', '_with_emb.parquet'))
        
        create_and_save_embeddings(
            df=df,
            emb_file_name=emb_file_name,
            texts_file_name=texts_file_name,
            df_file_name=df_file_name
        )
    

def load_dataframe(dataset_cfg: DatasetConfig):
    splits = ['train', 'dev', 'test', 'test_unseen']
    df_dict = {}
    for s in splits:
        split_cfg = getattr(dataset_cfg, s)
        if split_cfg is not None:
            file_name = getattr(split_cfg, 'df_file')
            if file_name is None:
                raise ValueError(f"{s} split is missing df_file in dataset config.")
            df_dict[s] = pd.read_json(os.path.join(dataset_cfg.dts_path or '', file_name), lines=True)
    return df_dict    
    
    
def create_and_save_embeddings(df: pd.DataFrame, emb_file_name: str, texts_file_name: str, df_file_name: str):
    df_emb, unique_texts, emb = create_embedding_openai(df)
    
    print(f"Saving embeddings to {emb_file_name}, texts to {texts_file_name}, df to {df_file_name}")
    save(
        emb_file_name=emb_file_name,
        texts_file_name=texts_file_name,
        df_file_name=df_file_name,
        emb=emb,
        texts=unique_texts,
        df=df_emb
    )
        
def create_embedding_openai(df: pd.DataFrame):
    load_dotenv()  # take environment variables from .env file
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    model = "text-embedding-3-small"
    all_texts = pd.concat([df["prompt"], df["chosen"], df["rejected"]], ignore_index=True)
    unique_texts = (all_texts.dropna().astype(str).unique().tolist())
    
    batch_size = 100        
    embeddings = []
    
    for i in tqdm(range(0, len(unique_texts), batch_size)):
        batch_texts = unique_texts[i:i+batch_size]
        
        resp = client.embeddings.create(
            input = batch_texts,
            model=model
        )
        embeddings.extend(emb.embedding for emb in resp.data)
    emb = np.array(embeddings, dtype=np.float32)
    
    text2idx = {t: i for i, t in enumerate(unique_texts)}

    df["prompt_emb_idx"]   = df["prompt"].astype(str).map(text2idx)
    df["chosen_emb_idx"]   = df["chosen"].astype(str).map(text2idx)
    df["rejected_emb_idx"] = df["rejected"].astype(str).map(text2idx)
    
    return df, unique_texts, emb

def save(emb_file_name: str, emb: np.ndarray, texts_file_name: str, texts: list, df_file_name: str, df: pd.DataFrame):
    np.save(emb_file_name, emb)

    with open(texts_file_name, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    df.to_parquet(df_file_name, index=False)
    
if __name__ == '__main__':
    main(sys.argv[1:])
