from typing import Optional
from pydantic import BaseModel
import numpy as np
import pandas as pd

class SplitFiles(BaseModel):
    df_file: Optional[str] = None
    text_file: Optional[str] = None
    emb_file: Optional[str] = None

class DatasetConfig(BaseModel):
    
    dts_name: str
    dts_path: Optional[str] = None
        
    train: SplitFiles
    dev: Optional[SplitFiles] = None
    test: Optional[SplitFiles] = None    
    test_unseen: Optional[SplitFiles] = None    
    
    prefix_columns: Optional[list[str]] = None
    
    @property
    def arch(self):
        return "Dataset config"