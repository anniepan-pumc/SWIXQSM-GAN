#!/usr/bin/env python
"""Train SWIXQSM-GAN from paired SWI/QSM NIfTI data."""

import argparse
import os
from collections import OrderedDict
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms

from data.ukb_loader import UKBDataset
from model.discriminator import Discriminator_ccgan
from model.unet import UNetSmall
from utils.train_gan import fit2d
from utils.weight_init import winit_func


def select_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_state_dict_handle_module(model, ckpt_path, strict=True):
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    clean_state_dict = OrderedDict()
    for key, value in state_dict.items():
        clean_state_dict[key.replace("module.", "")] = value

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=strict)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)
    return model


def build_transform(image_size):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size), interpolation=Image.Resampling.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train SWIXQSM-GAN on paired SWI/QSM slices.")
    parser.add_argument("--data_path", required=True, help="Root containing subject folders.")
    parser.add_argument("--record_path", required=True, help="CSV with SWI,QSM,InBoth columns.")
    parser.add_argument("--swi_subdir", default="", help="Optional SWI subdirectory under data_path.")
    parser.add_argument("--qsm_subdir", default="", help="Optional QSM subdirectory under data_path.")
    parser.add_argument("--output_dir", default=None, help="Checkpoint/output directory.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr_gen", type=float, default=1e-4)
    parser.add_argument("--lr_disc", type=float, default=1e-4)
    parser.add_argument("--content_weight", type=float, default=0.2)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--full_volume", action="store_true", help="Train on all slices per subject batch.")
    parser.add_argument("--gen_checkpoint", default=None, help="Optional generator checkpoint.")
    parser.add_argument("--disc_checkpoint", default=None, help="Optional discriminator checkpoint.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--fid_feature", type=int, default=192, choices=[64, 192, 768, 2048])
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device(args.device)
    use_amp = device.type == "cuda"
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 480 if device.type == "cuda" else 16 if device.type == "mps" else 8

    output_dir = args.output_dir or os.path.join("save", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    transform = build_transform(args.image_size)
    dataset = UKBDataset(
        data_path=args.data_path,
        record_path=args.record_path,
        SWI_path=args.swi_subdir,
        QSM_path=args.qsm_subdir,
        transform=transform,
        rand_slice=not args.full_volume,
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=args.shuffle, drop_last=True)

    generator = UNetSmall(num_channels=1, num_classes=1).to(device)
    discriminator = Discriminator_ccgan([1, args.image_size, args.image_size]).to(device)

    if args.gen_checkpoint:
        load_state_dict_handle_module(generator, args.gen_checkpoint)
    else:
        winit_func(generator)

    if args.disc_checkpoint:
        load_state_dict_handle_module(discriminator, args.disc_checkpoint, strict=False)
    else:
        winit_func(discriminator)

    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr_gen, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr_disc, betas=(0.5, 0.999))
    scheduler_g = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_g, 10, 1)
    scheduler_d = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_d, 10, 1)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Subjects/slices: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Output: {output_dir}")

    history = fit2d(
        generator,
        discriminator,
        train_loader,
        args.epochs,
        scheduler_g,
        scheduler_d,
        optimizer_g,
        optimizer_d,
        scaler,
        nn.L1Loss().to(device),
        nn.BCEWithLogitsLoss().to(device),
        device,
        FrechetInceptionDistance(feature=args.fid_feature),
        output_dir,
        rand_slice=not args.full_volume,
        use_amp=use_amp,
        grad_accum_steps=args.grad_accum_steps,
        content_w=args.content_weight,
    )

    result = pd.DataFrame(
        {
            "gen_loss": history[0],
            "disc_loss": history[1],
            "fid_qsm_fake": history[2],
            "fid_qsm_swi": history[3],
        }
    )
    result.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)


if __name__ == "__main__":
    main()
