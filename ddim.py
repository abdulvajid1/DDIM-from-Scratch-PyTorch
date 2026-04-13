import torch

class DDIM:
    def __init__(self, n_timesteps: int, st_beta: float, end_beta: float, device='cpu') -> None:
        self.device = device
        self.n_timesteps = n_timesteps
        self.betas = self.get_sheduler(st_beta, end_beta)
        self.alphas = 1 - self.betas
        
        self.cum_alphas = torch.cumprod(self.alphas)
        self.one_minus_cum_alphas = 1 - self.cum_alphas
        
        self.sq_cum_alphas = torch.sqrt(self.cum_alphas)
        self.sq_one_minus_cum_alphas = torch.sqrt(self.one_minus_cum_alphas)
        
        
    def forward_diffusion_sample(self, img: torch.Tensor, t: torch.Tensor):
        sq_cum_alpha = self.sq_cum_alphas[t] # get sq cum alphas of timesteps given
        sq_one_minus_cum_alpha = self.sq_one_minus_cum_alphas[t]
        
        # img size will be (b, c, h, w), alphas will be 1D, convert alphas view so each item in a batch will get each alphas
        sq_cum_alpha = sq_cum_alpha.view(-1, 1, 1, 1)
        sq_one_minus_cum_alpha = sq_one_minus_cum_alpha.view(-1, 1, 1, 1)
        
        noise = torch.randn_like(img) 
        
        xt = sq_cum_alpha * img + sq_one_minus_cum_alpha * noise
        
        return xt, noise
    
    def sample_timestep(self, n):
        return torch.randint(low=0, high=self.n_timesteps, device=self.device, size=(n,))
    
    def get_sheduler(self, st_beta=1e-4, end_beta=0.02, n_timesteps=1000, type='linear'):
        if type=='linear':
            return torch.linspace(st_beta, end_beta, n_timesteps)
        elif type=='cosine':
            pass
        
    def sample_image(self, n, n_timesteps):
        pass