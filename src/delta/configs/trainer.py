from typing import Optional, Tuple
from pydantic import BaseModel


class TrainerConfig(BaseModel):
    # Train time
    max_epochs: int = 100
    max_steps: Optional[int] = None
    sampler: Optional[str] = None
    batch_size: int = 32
    
    scheduler: str = "none"
    
    num_workers: int = 2
    

    # Optimizer
    optimizer: str = "Adam"  # [Adam, RMSprop, SGD]
    learning_rate: float #= 1e-3
    cyclic_lr: bool = False
    weight_decay: float = 0.0
    
    betas: Tuple[float, float]
    eps: float = 1e-8
    
    warmup_steps: Optional[int] = -1  # if -1, will use 5% of total steps
    min_lr: Optional[float] = 1e-6
    
    #second_optimizer: str = "SGD"
    #second_learning_rate: float = 5e-4
    #second_optimizer_start_epoch: Optional[int] = 75
    
    

    #freeze_bias_after: Optional[int] = None

    #ckpt_savedir: str = "./checkpoints/irt"
    
    #c_reg_s: float = 1e-6
    #c_reg_d: float = 1e-6