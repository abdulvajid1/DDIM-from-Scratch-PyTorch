from dataclasses import dataclass

@dataclass
class Arguments():
    n_timesteps: int = 1000
    batch_size: int = 64
    grad_accumulation_steps:int = 2
    n_epoch: int = 19000
    st_beta: float = 1e-4
    end_beta: float = 0.02
    time_dim: int = 128
    learning_rate: float = 1e-4
    l2_norm: float = 0.0
    eval_step: int =  100
    img_size: int = 64
    dataset_path: str ='data'
    eval_datasetpath: str ='data'
    sampling_steps: int = 50
    n_samples: int = 4
    
    