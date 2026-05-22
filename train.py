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
from utils import setup_logging, save_model, load_model
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import logging
import tqdm
import copy
from pathlib import Path

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                    level=logging.INFO, 
                    datefmt="%I: %M: %S")

# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.conv.fp32_precision = 'tf32'
torch.backends.cudnn.fp32_precision = "tf32"
torch.backends.cudnn.fp32_precision = "tf32"

import mlflow

mlflow.set_experiment("Deep Learning Experiment")

mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_sampling_interval(1)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------
class EMA:
    """
    Maintains a shadow (EMA) copy of model parameters.

    After each optimizer step call `ema.update(model)`. For evaluation /
    sampling, wrap the block with `ema.apply_shadow(model)` and then
    `ema.restore(model)` to put the original weights back.

    Args:
        model   : the model whose parameters we want to track
        decay   : EMA decay rate (typical values: 0.995 – 0.9999).
                  Higher → slower update, smoother weights.
        warmup_steps : number of optimizer steps before EMA starts collecting.
                       Before this threshold the shadow is simply the current
                       weights (effective decay = 0).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 100):
        self.decay        = decay
        self.warmup_steps = warmup_steps
        self.step         = 0

        # Deep-copy the initial weights as the starting shadow
        self.shadow: dict[str, torch.Tensor] = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        self._backup: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    def _effective_decay(self) -> float:
        """Ramp the decay from 0 → self.decay during warm-up."""
        if self.step < self.warmup_steps:
            return 0.0                       # shadow == current weights
        return self.decay

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Call once per optimizer step (after `optimizer.step()`)."""
        self.step += 1
        d = self._effective_decay()
        for name, param in model.named_parameters():
            self.shadow[name].mul_(d).add_(param.data, alpha=1.0 - d)

    # ------------------------------------------------------------------
    def apply_shadow(self, model: nn.Module) -> None:
        """Replace model weights with EMA shadow weights (saves originals)."""
        self._backup = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    # ------------------------------------------------------------------
    def restore(self, model: nn.Module) -> None:
        """Restore the original weights that were saved by `apply_shadow`."""
        for name, param in model.named_parameters():
            param.data.copy_(self._backup[name])
        self._backup = {}

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "step": self.step, "decay": self.decay}

    def load_state_dict(self, state: dict) -> None:
        self.shadow = state["shadow"]
        self.step   = state["step"]
        self.decay  = state["decay"]


# ---------------------------------------------------------------------------
# Helpers to persist / restore EMA alongside the model checkpoint
# ---------------------------------------------------------------------------
def save_model_with_ema(model: nn.Module, optimizer: AdamW, ema: EMA,
                        global_step: int, run_name: str) -> None:
    """Save model, optimizer, and EMA state in a single checkpoint."""
    save_model(model, optimizer, global_step=global_step, run_name=run_name)

    ema_path = f"model/ema_{global_step}.pt"
    # Path(ema_path).mkdir(exist_ok=True)
    torch.save(ema.state_dict(), ema_path)
    logging.info(f"EMA state saved → {ema_path}")


def load_ema(ema: EMA, args: Arguments, global_step: int) -> None:
    """Load EMA state from a checkpoint (best-effort; logs warning on miss)."""
    ema_path = f"model/ema_{global_step}.pt"
    try:
        state = torch.load(ema_path, map_location=device)
        ema.load_state_dict(state)
        logging.info(f"EMA state loaded ← {ema_path}")
    except FileNotFoundError:
        logging.warning(f"No EMA checkpoint found at {ema_path}; starting fresh EMA.")


# ---------------------------------------------------------------------------
# Evaluation  (uses EMA shadow weights for sampling)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def eval(ddim: DDIM, model: UNet, ema: EMA, loader: DataLoader,
         device: str, global_step: int, args: Arguments):
    model.eval()
    total_loss = 0
    i = 0

    for img, _ in loader:
        timesteps = ddim.sample_timestep(img.shape[0])
        img       = img.to(device)
        timesteps = timesteps.to(device)

        xt, real_noise = ddim.forward_diffusion_sample(img, timesteps)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            pred_noise = model(xt, timesteps)
            loss = F.mse_loss(real_noise, pred_noise)

        total_loss += loss.item()
        i += 1
        if i == 10:
            break

    avg_loss = total_loss / i

    # ---- sample with EMA weights ----------------------------------------
    ema.apply_shadow(model)          # swap in shadow weights
    sampled_imgs = ddim.sample_image(
        model, n=args.n_samples, n_steps=args.sampling_steps, eta=0.0
    )
    ema.restore(model)               # swap original weights back
    # ---------------------------------------------------------------------

    img_path = f"result/{global_step}.jpg"
    save_images(sampled_imgs, path=img_path)
    mlflow.log_artifact(img_path)

    model.train()
    return avg_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(ddim: DDIM, model: UNet, ema: EMA,
          train_loader: DataLoader, val_loader: DataLoader,
          scheduler: LambdaLR, optimizer: AdamW,
          eval_step: int, device: str, epoch: int,
          args: Arguments, global_step: int):

    model.train()
    progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch: {epoch}")

    global_step_now = (epoch * len(train_loader)) if not global_step else global_step

    for step, (img, _) in enumerate(progress_bar):
        global_step = global_step_now + step

        timesteps = ddim.sample_timestep(img.shape[0])
        img       = img.to(device)
        timesteps = timesteps.to(device)

        xt, real_noise = ddim.forward_diffusion_sample(img, timesteps)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            pred_noise = model(xt, timesteps)
            raw_loss   = F.mse_loss(real_noise, pred_noise)
            loss       = raw_loss / args.grad_accumulation_steps

        loss.backward()

        # Gradient accumulation
        if (step + 1) % args.grad_accumulation_steps == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            # ---- EMA update (once per true optimizer step) --------------
            ema.update(model)
            # -------------------------------------------------------------

        # Eval
        if (global_step + 1) % eval_step == 0:
            logging.info("Evaluating…")
            avg_val_loss = eval(ddim, model, ema, val_loader, device,
                                global_step=global_step, args=args)
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": raw_loss.item(),
                "eval_loss" : avg_val_loss,
                "ema_step"  : ema.step,
            }, step=global_step + 1)

            logging.info(
                f"Step: {global_step+1} | Loss: {raw_loss.item():.5f} "
                f"| Eval_Loss: {avg_val_loss:.5f} | EMA step: {ema.step}"
            )

        # Checkpoint
        if (global_step + 1) % args.save_step == 0:
            save_model_with_ema(model, optimizer, ema,
                                global_step=(global_step + 1), run_name="testing")

    return global_step


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
def main():
    args   = Arguments()
    ddim      = DDIM(device=device, args=args)
    model     = UNet(c_in=3, time_dim=args.time_dim, device=device, channel_mults=args.channel_multiplier, base_channels=args.base_channel).to(device)
    # model     = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate, weight_decay=args.l2_norm)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    # ---- EMA (decay=0.9999, warm-up for the first 100 optimiser steps) --
    ema = EMA(model, decay=args.ema_decay, warmup_steps=args.ema_warmup)
    # ---------------------------------------------------------------------

    global_step = None
    if args.load_model:
        global_step = load_model(model, optimizer, args)
        # Try to reload a matching EMA checkpoint
        if global_step is not None:
            load_ema(ema, args, global_step=global_step)

    logging.info("Initialized Model, Optimizer & EMA")

    train_loader = get_dataloader(args, train=True, single_batch=False)
    val_loader   = train_loader
    logging.info("DataLoader setup complete")

    params = {
        "micro_batch_size"      : args.batch_size,
        "gradient_accm_steps"   : args.grad_accumulation_steps,
        "macro_batch_size"      : args.batch_size * args.grad_accumulation_steps,
        "learning_rate"         : args.learning_rate,
        "l2_norm"               : args.l2_norm,
        "time_emb_dim"          : args.time_dim,
        "image_size"            : args.img_size,
        "channel_multiplier"    : args.channel_multiplier,
        "ema_decay"             : args.ema_decay,
        "ema_warmup_steps"      : args.ema_warmup,
        "base_channels"         : args.base_channel,
        "channel_mults"         : args.channel_multiplier
    }

    with mlflow.start_run():
        mlflow.log_params(params=params)

        for epoch in range(args.n_epoch):
            global_step = train(
                ddim, model, ema,
                train_loader, val_loader,
                scheduler, optimizer,
                eval_step=args.eval_step,
                device=device,
                epoch=epoch,
                args=args,
                global_step=global_step,
            )


if __name__ == "__main__":
    main()