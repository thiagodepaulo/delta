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
        self.r_star_model.eval()
        for p in self.r_star_model.parameters():
            p.requires_grad = False
        
        print(argparse.Namespace(**trainer_config.model_dump()))            
        self.save_hyperparameters(argparse.Namespace(**trainer_config.model_dump()))
        
    def compute_loss(self, batch):
        
        u, x, y_w, y_l = batch['u_id'].long(), batch['prompt_emb'], batch['chosen_emb'], batch['rejected_emb']
        text_x = batch['prompt']
        text_y_w = batch['chosen']
        text_y_l = batch['rejected']
        user_features = batch.get('features', None)
        
        with torch.no_grad():
            out_r_star_w = self.r_star_model(text_x, text_y_w)
            out_r_star_l = self.r_star_model(text_x, text_y_l)
        
        out_delta_w = self.delta_model(user_u=u, answer_y=y_w, prompt_x=x, user_features=user_features)
        out_delta_l = self.delta_model(user_u=u, answer_y=y_l, prompt_x=x, user_features=user_features)
        
        diff_r_star = out_r_star_w - out_r_star_l
        diff_delta = out_delta_w["logits"] - out_delta_l["logits"]
        z = diff_r_star + diff_delta
        
        bt_loss = F.softplus(-z).mean()  
        
        total_reg_loss, log_reg = self.delta_model.compute_regularization()
        
        # total loss
        loss = bt_loss + total_reg_loss
        
        # accuracy
        acc = (z > 0).float().mean()
        acc_r_star = (diff_r_star > 0).float().mean()
        acc_delta = (diff_delta > 0).float().mean()
        
        log = {       
            "loss": loss.mean(),                 
            "bt_loss": bt_loss.mean(),
            "reg_loss": total_reg_loss.mean(),
            "acc": acc,
            "acc_r_star": acc_r_star,
            "acc_delta": acc_delta,
            **log_reg
        } 
        
        return loss.mean(), log
    
    def training_step(self, batch):        
        loss, log = self.compute_loss(batch)                
        self._log_split("train", log)        
        return loss
    
    def validation_step(self, batch):        
        loss, log = self.compute_loss(batch)
        self._log_split("val", log)        
        return loss
    
    def on_train_epoch_start(self):
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        print(f"Epoch {self.current_epoch} | LR = {lr:.6e}")

    
    def _log_split(self, split: str, log: Dict[str, torch.Tensor]):
        self.log(f"{split}_loss", log["loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{split}_acc", log["acc"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{split}_acc_r_star", log["acc_r_star"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{split}_acc_delta", log["acc_delta"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{split}_bt_loss", log["bt_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{split}_reg_loss", log["reg_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def on_test_epoch_start(self):
        # 0 -> test, 1 -> test_unseen
        self.test_acc_sum = [torch.tensor(0.0, device=self.device),
                             torch.tensor(0.0, device=self.device)]
        self.test_n = [torch.tensor(0.0, device=self.device),
                       torch.tensor(0.0, device=self.device)]
            
    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        loss, log = self.compute_loss(batch)                 
        
       # log separado por split
        split_name = "test" if dataloader_idx == 0 else "test_unseen"
        self.log(f"{split_name}_loss", log["loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{split_name}_acc", log["acc"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return DataLoader(self.test_seen_dataset, batch_size=self.args.batch_size, shuffle=False)
        return DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False)
    
    def test_dataloader(self):
        test_loaders = [DataLoader(self.test_seen_dataset, batch_size=self.args.batch_size, shuffle=False)]
        if self.test_unseen_dataset is not None:
            test_loaders.append(DataLoader(self.test_unseen_dataset, batch_size=self.args.batch_size, shuffle=False))
        return test_loaders


if __name__ == "__main__":
    lit_module = PrefModule()    
    
    