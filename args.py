from dataclasses import dataclass

@dataclass
class Arguments():
    n_timesteps: int = 1000
    batch_size: int = 8
    n_epoch: int = 10
    st_beta: float = 1e-4
    end_beta: float = 0.02
    time_dim: int = 128
    learning_rate: float = 1e-3
    l2_norm: float = 0.01
    eval_step: int =  1
    img_size: int = 64
    dataset_path: str ='data'
    eval_datasetpath: str ='data'
    
    