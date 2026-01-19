

def load(emb_file_name: str, texts_file_name: str, df_file_name: str):
    emb = np.load(emb_file_name)
    texts = json.load(open(texts_file_name))
    if df_file_name.endswith('.parquet'):
        df = pd.read_parquet(df_file_name)
    elif df_file_name.endswith('.jsonl'):
        df = pd.read_json(df_file_name, lines=True)    
    return {"embeddings": emb, "texts": texts, "df": df}

def load_from_dataset(dataset_name: str, dataset_config_file: str = '', splits = ['train', 'dev', 'test']):
    if dataset_config_file == '':
        dataset_config_file = DEFAULT_DTS_CONFIG_FILE
    with open(dataset_config_file, "r") as f:
        dc_ = yaml.safe_load(f)
        dataset_cfg = DatasetConfig(**dc_[dataset_name])

    results = {}
    for split in splits:        
        emb_file_name = os.path.join(dataset_cfg.dts_path or '', getattr(dataset_cfg, split).emb_file)
        texts_file_name = os.path.join(dataset_cfg.dts_path or '', getattr(dataset_cfg, split).text_file)
        df_file_name = os.path.join(dataset_cfg.dts_path or '', getattr(dataset_cfg, split).df_file)
        results[split] = load(emb_file_name, texts_file_name, df_file_name)
    
    return results