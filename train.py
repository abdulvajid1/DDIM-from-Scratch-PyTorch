import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from utils import get_dataloader

from args import Arguments
from ddim import DDIM

from unet import UNet

from utils import save_images
from utils import setup_logging

@torch.inference_mode()
def eval(ddim: DDIM, model:UNet, loader: DataLoader, device: str, global_step: int):
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
    
    sampled_imgs = ddim.sample_image(model, n=4)
    save_images(sampled_imgs, path=f'result/{global_step}.jpg')
    
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
        print((step + 1) % eval_step == 0)
        print(step, eval_step)
        if (step+1) % eval_step == 0:
            
            print('Evaluting.......')
            model.eval()
            avg_val_loss = eval(ddim, model, val_loader, device, global_step=((epoch+1)* xt.shape[0]) + step)
            print(f"Epoch: {epoch} | Step: {step+1} | Loss: {loss.item()} | Eval_Loss: {avg_val_loss}")
            model.train()
             
        

def main():
    args = Arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # setup_logging(run_name=f'{args.n_batchsize}-{args.st_beta}-{args.learning_rate}-{args.time_dim}-{args.l2_norm}-{args.img_size}')    
    
    ddim = DDIM(device=device, args=args)
    model = UNet(c_in=3, time_dim=args.time_dim, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_norm)
    
    train_loader = get_dataloader(args)
    val_loader = get_dataloader(args=args, train=False)
    
    for epoch in range(args.n_epoch):
        train(ddim, model, train_loader, val_loader, optimizer, eval_step=args.eval_step, device=device, epoch=epoch)
    
    
if __name__ == '__main__':
    main()