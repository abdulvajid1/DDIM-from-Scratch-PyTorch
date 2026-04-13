from dataclasses import dataclass

@dataclass
class Arguments():
    n_timesteps: int = 1000
    n_batchsize: int = 8
    n_epoch: int = 10
    st_beta: float = 1e-4
    end_beta: float = 0.02
    time_dim: int = 128
    learning_rate: float = 1e-3
    l2_norm: float = 0.01
    eval_step: int =  10
    
    