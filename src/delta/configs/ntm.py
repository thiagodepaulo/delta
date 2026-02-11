from typing import Optional, Tuple
from pydantic import BaseModel

class NTMConfig(BaseModel):    
    
    
    vocab_size: Optional[int] = None
    words_emb_dim: int = 300
    n_topics: int
    n_labels: int
    n_prior_covars: int
    n_topic_covars: int
    classifier_layers: int
    use_interactions: bool
    l1_beta_reg: float
    l1_beta_c_reg: float
    l1_beta_ci_reg: float
    l2_prior_reg: float
    classify_from_covars: bool = True
    user_covars: bool = True
    topic_covar_names: Optional[list[str]] = None
    eta_bn_prop = Optional[int] = 1
    alpha = Optional[int] = 1
    
        
    @property
    def arch(self):
        return "Neural Topic Model"
    


    
        
        