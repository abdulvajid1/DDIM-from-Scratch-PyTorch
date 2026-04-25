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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import tqdm

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                    level=logging.INFO, 
                    datefmt="%I: %M: %S")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.conv.fp32_precision = 'tf32'
torch.backends.cudnn.fp32_precision = "tf32"      # cuDNN globally → tf32




@torch.inference_mode()
def eval(ddim: DDIM, model:UNet, loader: DataLoader, device: str, global_step: int, args: Arguments):
    model.eval()
    total_loss = 0
    i = 0
    for img, _ in loader:
        # forward diffusion, add noise in a random timestep
        timesteps = ddim.sample_timestep(img.shape[0]) # get batchsize dynamically
        
        img       = img.to(device)
        timesteps = timesteps.to(device)
        
        # Forward
        xt, real_noise = ddim.forward_diffusion_sample(img, timesteps)


        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            # predict noise added
            pred_noise = model(xt, timesteps)
            loss = F.mse_loss(real_noise, pred_noise)
        
        total_loss += loss.item()
        i += 1

        if i==31:
            break
    
    avg_loss = total_loss/i
    
    sampled_imgs = ddim.sample_image(model, n=args.n_samples, n_steps=args.sampling_steps)
    save_images(sampled_imgs, path=f'result/{global_step}.jpg')
    
    model.train()
    return avg_loss
    

def train(ddim: DDIM, model: UNet, train_loader: DataLoader, val_loader: DataLoader, sheduler: ReduceLROnPlateau, optimizer: AdamW, eval_step: int, device:str, epoch: int, args: Arguments):
    model.train()
    progress_bar = tqdm.tqdm(train_loader, desc=f'Epoch: {epoch}')
    total_steps_in_epoch = (epoch * len(train_loader))

    for step, (img, _) in enumerate(progress_bar):
        global_step = total_steps_in_epoch + step
        timesteps = ddim.sample_timestep(img.shape[0]) # forward diffusion, add noise in a random timestep
        
        img = img.to(device)
        timesteps = timesteps.to(device)
        xt, real_noise = ddim.forward_diffusion_sample(img, timesteps)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            pred_noise = model(xt, timesteps) # predict noise added
            raw_loss = F.mse_loss(real_noise, pred_noise)
            loss = raw_loss / args.grad_accumulation_steps

        loss.backward()

        # Update only each even numbers or for the final step
        if (step+1) % args.grad_accumulation_steps == 0 or (step + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        if (global_step+1) % eval_step == 0:          
            logging.info('Evaluating....')
            avg_val_loss = eval(ddim, model, val_loader, device, global_step=global_step, args=args)
            sheduler.step(avg_val_loss)
            logging.info(f"Step: {global_step+1} | Loss: {raw_loss.item()} | Eval_Loss: {avg_val_loss}")
            
        

        

def main():
    args = Arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # setup_logging(run_name=f'{args.n_batchsize}-{args.st_beta}-{args.learning_rate}-{args.time_dim}-{args.l2_norm}-{args.img_size}')    
    
    ddim = DDIM(device=device, args=args)
    model = UNet(c_in=3, time_dim=args.time_dim, device=device).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_norm)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=100)
    logging.info('Initialized Model & Optimizer')

    train_loader = get_dataloader(args, train=True, single_batch=False)
    val_loader = train_loader
    logging.info('DataLoader Setup Completed')

    for epoch in range(args.n_epoch):
        train(ddim, model, train_loader, val_loader, scheduler, optimizer,  eval_step=args.eval_step, device=device, epoch=epoch, args=args)
    
    
if __name__ == '__main__':
    main()