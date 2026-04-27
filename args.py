from dataclasses import dataclass
import torch

@dataclass
class Arguments():
    n_timesteps: int = 1000
    batch_size: int = 16
    grad_accumulation_steps:int = 16
    n_epoch: int = 19000
    st_beta: float = 1e-4
    end_beta: float = 0.02
    time_dim: int = 128
    learning_rate: float = 1e-4
    l2_norm: float = 0.0
    eval_step: int =  2500
    img_size: int = 128
    dataset_path: str ='data'
    eval_datasetpath: str ='data'
    sampling_steps: int = 500
    n_samples: int = 4
    channel_multiplier: int = 2
    save_step: int = 5000
    load_model: bool = False
    device: str = 'cuda' if torch.cuda.is_available() else "cpu"
    
    