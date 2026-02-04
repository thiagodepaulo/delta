from typing import Optional, Tuple
from pydantic import BaseModel

class NTMConfig(BaseModel):    
    
    vocab_size: int
    words_emb_dim: int
    n_topics: int
    n_labels: int
    n_prior_covars: int
    n_topic_covars: int
    classifier_layers: Tuple[int, ...]
    use_interactions: bool
    l1_beta_reg: float
    l1_beta_c_reg: float
    l1_beta_ci_reg: float
    l2_prior_reg: float
    classify_from_covars: bool = False
        
    @property
    def arch(self):
        return "Neural Topic Model"
    


    
        
        