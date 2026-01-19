import sys
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from delta.lit_module import PrefModule, PrefDataModule
from delta.utils.config_utils import load_config
from delta.configs.trainer import TrainerConfig
from delta.reward_models.zero_rw import ZeroRWModel
from delta.configs.caimira import CaimiraConfig
from delta.models.caimira import CaimiraModel
from delta.models.model1 import Model1
from delta.configs.model1 import Model1Config

def main(args):
    parser = argparse.ArgumentParser(description="Train a model")    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--dts_name", type=str, required=True, help="Dataset name")    
    parser.add_argument("--dts_config_file", type=str, required=False, default='/home/thiagodepaulo/exp/delta/configs/data_configs.yaml', help="Dataset Config file name")
    parser.add_argument("--config_file", type=str, default='/home/thiagodepaulo/exp/delta/configs/exp_config.yaml', help="YAML config file")
    parser.add_argument("--model_name", type=str, default='model1', help="Model name to use")
    parser.add_argument("--map_rw_dataset", type=str, default=None, help="Map reward model dataset name")
    parser.add_argument("--exp_name", type=str, default="default_exp", help="Experiment name for logging")
    parser.add_argument("--exp_version", type=str, default="v0", help="Experiment version for logging")
    parser.add_argument("--features", type=str, default=None, help="Feature columns to use, 'all' for all features")
    parser.add_argument("--patience", type=int, default=10**9, help="Early stopping patience")
    parser.add_argument("--n_dim", type=int, default=50, help="Number of dimensions")
    args = parser.parse_args()
    
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=args.exp_name,
        version=args.exp_version
    )                
            
    config_dict = load_config(args.config_file)
    trainer_config = TrainerConfig(**config_dict["trainer"])    
    
    if args.map_rw_dataset is not None:
        rwModel = MapRWModel(args.map_rw_dataset)
    else:
        rwModel = ZeroRWModel()
        
    if args.model_name == 'model1':
        model1_config = Model1Config(**config_dict["model1"])    
        model1_config.has_user_features = True if args.features is not None else False
        model1_config.n_dim_user_features = 45 if args.features == "all" else 0        
        model1_config.n_dim = args.n_dim
        model_instance = Model1(model1_config)
    elif args.model_name == 'caimira':
        caimira_config = CaimiraConfig(**config_dict["caimira"])
        caimira_config.n_dim = args.n_dim
        model_instance = CaimiraModel(caimira_config)  
    model = PrefModule(trainer_config, model_instance, rwModel)
    
    early_stop = EarlyStopping(
        monitor="val_loss",   # metric name
        patience=args.patience,           # epochs with no improvement
        mode="min",           # "min" for loss, "max" for accuracy
    )

    data_module = PrefDataModule(args)
    
    trainer = L.Trainer(
        max_epochs=trainer_config.max_epochs,
        logger=logger,
        callbacks=[early_stop]
    ) 
    trainer.fit(model, datamodule=data_module)
        
    trainer.test(model, datamodule=data_module)
    

        
    
if __name__ == "__main__":
    main(sys.argv[1:])