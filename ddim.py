import torch
import math
from args import Arguments

class DDIM:
    def __init__(self, args: Arguments, device='cpu') -> None:
        self.img_size = args.img_size
        self.device = device
        self.n_timesteps = args.n_timesteps
        self.betas = self.get_sheduler(args.st_beta, args.end_beta, args.n_timesteps, type='linear')
        self.alphas = 1 - self.betas
        
        self.cum_alphas = torch.cumprod(self.alphas, dim=-1)
        self.one_minus_cum_alphas = 1 - self.cum_alphas
        
        self.sq_cum_alphas = torch.sqrt(self.cum_alphas).to(device=device)
        self.sq_one_minus_cum_alphas = torch.sqrt(self.one_minus_cum_alphas).to(device)
        
        
    def forward_diffusion_sample(self, img: torch.Tensor, t: torch.Tensor):
        sq_cum_alpha = self.sq_cum_alphas[t] # get sq cum alphas of timesteps given
        sq_one_minus_cum_alpha = self.sq_one_minus_cum_alphas[t]
        
        # img size will be (b, c, h, w), alphas will be 1D, convert alphas view so each item in a batch will get each alphas
        sq_cum_alpha = sq_cum_alpha.view(-1, 1, 1, 1)
        sq_one_minus_cum_alpha = sq_one_minus_cum_alpha.view(-1, 1, 1, 1)
        noise = torch.randn_like(img, device=self.device)
    
        xt = sq_cum_alpha * img + sq_one_minus_cum_alpha * noise

        return xt, noise
    
    def sample_timestep(self, n):
        return torch.randint(low=0, high=self.n_timesteps, device=self.device, size=(n,))
    
    def get_sheduler(self, st_beta=1e-4, end_beta=0.02, n_timesteps=1000, type='linear'):
        if type=='linear':
            return torch.linspace(st_beta, end_beta, n_timesteps).to(device=self.device)
        elif type=='cosine':
            pass
    
    @torch.inference_mode()    
    def sample_image(self, model, n, n_steps: int = 1000, eta: float = 0.0):

        timesteps = torch.linspace(0, self.n_timesteps - 1, n_steps).long().flip(0)  # [999, 949, ..., 0]
        xt = torch.randn(n, 3, self.img_size, self.img_size).to(self.device)

        for step, i in enumerate(timesteps):
            i = i.item()

            # Previous timestep in the schedule (next in reversed order)
            i_prev = timesteps[step + 1].item() if step + 1 < len(timesteps) else 0

            curr_timesteps = torch.tensor([i] * n).to(self.device)

            # Predict noise and reconstruct x0
            pred_noise = model(xt, curr_timesteps)
            pred_x0 = (xt - self.sq_one_minus_cum_alphas[i] * pred_noise) / self.sq_cum_alphas[i]
            pred_x0 = pred_x0.clamp(-1, 1)  # optional but recommended

            if i == 0:
                return pred_x0

            # Correct DDIM sigma
            sigma = eta * torch.sqrt(
            (1 - self.cum_alphas[i_prev]) / (1 - self.cum_alphas[i]) *
            (1 - self.cum_alphas[i] / self.cum_alphas[i_prev])
            )

            # DDIM update step
            direction = (self.one_minus_cum_alphas[i_prev] - sigma ** 2) ** 0.5 * pred_noise
            new_noise = torch.randn_like(xt).to(self.device)
            xt = self.sq_cum_alphas[i_prev] * pred_x0 + direction + sigma * new_noise

        return xt
            