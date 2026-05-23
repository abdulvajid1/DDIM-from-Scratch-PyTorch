from dataclasses import dataclass
import torch

@dataclass
class Arguments():
    n_timesteps: int = 1000
    batch_size: int = 32
    grad_accumulation_steps:int = 2
    n_epoch: int = 19000
    st_beta: float = 1e-4
    end_beta: float = 0.02
    time_dim: int = 128
    learning_rate: float = 1e-4
    l2_norm: float = 0.01
    eval_step: int =  100
    img_size: int = 128
    dataset_path: str ='data'
    eval_datasetpath: str ='data'
    sampling_steps: int = 1000
    n_samples: int = 4
    channel_multiplier: tuple = (1, 2, 2, 2)
    save_step: int = 1000
    load_model: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else "cpu"
    ema_decay: float = 0.999
    ema_warmup: int = 1000
    base_channel : int = 192
    
    