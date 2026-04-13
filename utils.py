import os
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def plot_images(images):
    plt.figure(figsize=(10, 10))
    plt.imshow(torch.cat([
                torch.cat([i for i in images.cpu()], dim=-1)
            ], dim=-2).permute(1, 2, 0).to('cpu'))
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_dataloader(args, train=True):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.img_size),
        # torchvision.transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        # torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    if train:
        dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=6)
    else:
        dataset = torchvision.datasets.ImageFolder(args.eval_datasetpath, transform=transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=1)
    
    return dataloader

def save_model(model, optimizer, global_step, run_name):
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step
    }

    torch.save(obj, os.path.join("models", run_name, f"model_{global_step}.ckpt"))
    

def load_model(model, optimizer, global_step, args):
    checkpoint = torch.load(os.path.join('models', args.run_name, f"model_{global_step}.ckpt"), map_location=args.device)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    global_step = checkpoint.get("global_step", 0)

    print(f"Model loaded from model_{global_step}.ckpt (step {global_step})")

    return global_step

def setup_logging(run_name):
    os.makedirs('models', exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)