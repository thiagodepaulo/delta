import argparse
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from delta.configs.trainer import TrainerConfig
from delta.models.base import BaseDeltaModel
from delta.reward_models.base import BaseRewardModel
from delta.data.dataset import create_torch_dataset

class PrefModule(L.LightningModule):

    def __init__(self, trainer_config: TrainerConfig, delta_model: BaseDeltaModel, r_star_model: BaseRewardModel):
        super().__init__()                
        self.trainer_config = trainer_config
        self.delta_model = delta_model
        
        self.r_star_model = r_star_model
        #self.r_star_model.eval()
        for p in self.r_star_model.parameters():
            p.requires_grad = False
        
        print(argparse.Namespace(**trainer_config.model_dump()))            
        self.save_hyperparameters(argparse.Namespace(**trainer_config.model_dump()))
        
    def compute_loss(self, batch):
                
        batch_w = batch["chosen"]
        batch_l = batch["rejected"]                
        
        with torch.no_grad():
            out_r_star_w = self.r_star_model(batch_w)
            out_r_star_l = self.r_star_model(batch_l)
        diff_r_star = out_r_star_w - out_r_star_l
        
        out_delta_w = self.delta_model(batch_w)        
        out_delta_l = self.delta_model(batch_l)
                
        recon_loss = (out_delta_w['loss'] + out_delta_l['loss'])
        nl_loss = (out_delta_w['nl_loss'] + out_delta_l['nl_loss'])
        ll_loss = (out_delta_w['ll_loss'] + out_delta_l['ll_loss'])
        kld_loss = (out_delta_w['kld_loss'] + out_delta_l['kld_loss'])
                          
        diff_delta = out_delta_w["logits"] - out_delta_l["logits"]
        z = diff_r_star + diff_delta
        bt_loss = F.softplus(-z).mean()
        
        # total loss
        loss = (bt_loss + ll_loss) + (nl_loss + kld_loss)
              
        # accuracy
        acc = (z > 0).float().mean()
        acc_r_star = (diff_r_star > 0).float().mean()
        acc_delta = (diff_delta > 0).float().mean()
        
        log = {       
            "loss": loss,                       
            "bt_loss": bt_loss,
            "recon_loss": recon_loss,            
            "nl_loss": nl_loss,
            "ll_loss": ll_loss,
            "kld_loss": kld_loss,
            "acc": acc,
            "acc_r_star": acc_r_star,
            "acc_delta": acc_delta,            
        } 
        
        return loss.mean(), log
    
    def split_batch_for_pref(self, batch):
        batch_w = {}
        batch_l = {}
        for k, v in batch.items():
            if k.startswith("chosen_"):
                new_k = "answer_" + k[len("chosen_") :]
                batch_w[new_k] = v
            elif k.startswith("rejected_"):
                new_k = "answer_" + k[len("rejected_") :]
                batch_l[new_k] = v
            else:
                batch_w[k] = v
                batch_l[k] = v
        return {"chosen": batch_w, "rejected": batch_l}
    
    def training_step(self, batch):      
        batch_wl = self.split_batch_for_pref(batch)
        
        loss, log = self.compute_loss(batch_wl)    
        bs = batch["u_id"].shape[0] if isinstance(batch, dict) and "u_id" in batch else 1
        self._log_split("train", log, batch_size=bs)            
        return loss
    
    def validation_step(self, batch):        
        batch_wl = self.split_batch_for_pref(batch)
        loss, log = self.compute_loss(batch_wl)
        bs = batch["u_id"].shape[0] if isinstance(batch, dict) and "u_id" in batch else 1
        self._log_split("val", log, batch_size=bs)        
        return loss
    
    def on_train_epoch_start(self):
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        print(f"Epoch {self.current_epoch} | LR = {lr:.6e}")
    
    def _log_split(self, split: str, log: Dict[str, torch.Tensor], batch_size: Optional[int] = None):
        self.log(f"{split}_loss", log["loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log(f"{split}_acc", log["acc"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{split}_acc_r_star", log["acc_r_star"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{split}_acc_delta", log["acc_delta"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{split}_bt_loss", log["bt_loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)        
        self.log(f"{split}_recon_loss", log["recon_loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{split}_ll_loss", log["ll_loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{split}_nl_loss", log["nl_loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{split}_kld_loss", log["kld_loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_test_epoch_start(self):
        # 0 -> test, 1 -> test_unseen        
        self.test_acc_sum = [torch.tensor(0.0, device=self.device),
                             torch.tensor(0.0, device=self.device)]
        self.test_n = [torch.tensor(0.0, device=self.device),
                       torch.tensor(0.0, device=self.device)]
            
    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        batch_wl = self.split_batch_for_pref(batch)
        loss, log = self.compute_loss(batch_wl)                 
        
       # log separado por split
        split_name = "test" if dataloader_idx == 0 else "test_unseen"
        bs = batch["u_id"].shape[0] if isinstance(batch, dict) and "u_id" in batch else 1
        self.log(f"{split_name}_loss", log["loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log(f"{split_name}_acc", log["acc"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)

        # acumula média manual (ponderada por batch size)
        bs = batch["u_id"].shape[0] if isinstance(batch, dict) and "u_id" in batch else 1
        self.test_acc_sum[dataloader_idx] += log["acc"].detach() * bs
        self.test_n[dataloader_idx] += float(bs)
        
        return loss
    
    def on_test_epoch_end(self):
        # reduz entre GPUs
        acc_test = (
            self.all_gather(self.test_acc_sum[0]).sum()
            / torch.clamp(self.all_gather(self.test_n[0]).sum(), min=1.0)
        )
        acc_unseen = (
            self.all_gather(self.test_acc_sum[1]).sum()
            / torch.clamp(self.all_gather(self.test_n[1]).sum(), min=1.0)
        )

        # logs finais (médias)
        self.log("test_acc_avg", acc_test, prog_bar=True, sync_dist=True)
        self.log("test_unseen_acc_avg", acc_unseen, prog_bar=True, sync_dist=True)
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.delta_model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=tuple(self.hparams.betas),
            eps=self.hparams.eps,
        )        
        
        sched_name = str(self.hparams.scheduler).lower()
        if sched_name == "none":
            return opt
        
        if sched_name == "cosine":
            # Uses total training steps from Trainer (works well with Lightning)
            # Optionally do warmup manually via LambdaLR.
            total_steps = self.trainer.estimated_stepping_batches

            warmup_steps = getattr(self.hparams, "warmup_steps", -1)
            if warmup_steps < 0:
                warmup_steps = int(0.05 * total_steps)  # 5% warmup
            self.hparams.warmup_steps = warmup_steps
            def lr_lambda(step: int):
                if self.hparams.warmup_steps > 0 and step < self.hparams.warmup_steps:
                    return float(step) / float(max(1, self.hparams.warmup_steps))
                progress = (step - self.hparams.warmup_steps) / float(max(1, total_steps - self.hparams.warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
                # cosine from 1 -> min_lr/lr
                cos = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()
                min_ratio = float(self.hparams.min_lr) / float(self.hparams.learning_rate)
                return min_ratio + (1.0 - min_ratio) * cos

            sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

        if sched_name == "onecycle":
            max_lr = float(self.hparams.max_lr) if self.hparams.max_lr is not None else float(self.hparams.learning_rate)
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=max_lr,
                total_steps=self.trainer.estimated_stepping_batches,
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

        if sched_name == "reduce_on_plateau":
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=float(self.hparams.plateau_factor),
                patience=int(self.hparams.plateau_patience),
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}

        raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")
    
    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

class PrefDataModule(L.LightningDataModule):
    
    def __init__(self, args):
        super().__init__()
        self.args = args            
        
    def setup(self, stage: str):
        dts = create_torch_dataset(self.args)
        self.train_dataset = dts['train']
        self.test_seen_dataset = dts['test']
        self.val_dataset = None
        if 'dev' in dts.keys():
            self.val_dataset = dts['dev']        
        self.test_unseen_dataset = None
        if 'test_unseen' in dts.keys():
            self.test_unseen_dataset = dts['test_unseen']     
            
        if self.args.model_name == 'ntm': 
            from delta.data.dataset import load_vocab
        
            self.args.vocab = load_vocab(self.args.dts_config_file, self.args.dts_name)
            print(f"Loaded vocab with {len(self.args.vocab)} tokens for NTM model")
            
            self.args.vocab_size = len(self.args.vocab)
            self.args.n_features = dts['train'].n_features
            self.args.feature_columns = dts['train'].feature_columns
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return DataLoader(self.test_seen_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        return DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
    
    def test_dataloader(self):
        test_loaders = [DataLoader(self.test_seen_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)]
        if self.test_unseen_dataset is not None:
            test_loaders.append(DataLoader(self.test_unseen_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers))
        return test_loaders
        
        