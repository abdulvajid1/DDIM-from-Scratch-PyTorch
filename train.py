import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from args import Arguments
from ddim import DDIM

from unet import UNet

from utils import save_images

@torch.inference_mode()
def eval(ddim, model, loader, device):
    total_loss = 0
    i = 0
    for img, _ in loader:
        # forward diffusion, add noise in a random timestep
        timesteps = ddim.sample_timestep(img.shape[0]) # get batchsize dynamically
        
        img = img.to(device)
        timesteps = timesteps.to(device)
        
        xt, real_noise = ddim.forward_diffusion_sample(img, timesteps)
        # predict noise added
        pred_noise = model(xt, timesteps)
        loss = F.mse_loss(real_noise, pred_noise)
        
        total_loss += loss.item()
        i += 1
    
    avg_loss = total_loss/i
    
    return avg_loss
    

def train(ddim: DDIM, model: UNet, train_loader: DataLoader, val_loader: DataLoader, optimizer: AdamW, eval_step: int, device:str, epoch):
    model.train()
    for step, (img, _) in enumerate(train_loader):
        
        # forward diffusion, add noise in a random timestep
        timesteps = ddim.sample_timestep(img.shape[0])
        xt, real_noise = ddim.forward_diffusion_sample(img, timesteps)
        # predict noise added
        pred_noise = model(xt, timesteps)
        
        loss = F.mse_loss(real_noise, pred_noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step+1) % eval_step:
            model.eval()
            avg_val_loss = eval(ddim, model, val_loader, device)
            print(f"Epoch: {epoch} | Step: {step+1} | Loss: {loss.item()} | Eval_Loss: {avg_val_loss}")
            ddim.sample_image(model, n=4, n_timesteps=50)
            model.train()
            
       
            
        
        

def main():
    from utils import get_dataloader
    args = Arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ddim = DDIM(args.n_timesteps, args.st_beta, args.end_beta, device=device)
    model = UNet(c_in=3, time_dim=args.time_dim, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_norm)
    
    train_loader = get_dataloader(args)
    val_loader = get_dataloader(args=args, train=False)
    
    for i in range(args.n_epoch):
        train(ddim, model, train_loader, val_loader, optimizer, eval_step=args.eval_step, device=device, epoch=args.n_epoch)
    