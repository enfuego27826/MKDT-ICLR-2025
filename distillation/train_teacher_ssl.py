import os
import torch
import torch.nn as nn
from tqdm import tqdm

from dataloader.dataloader import get_ssl_dataloaders
from models.builder import build_model
from losses.barlow_twins import BarlowTwins
from utils.checkpoints import save_checkpoint
from utils.logger import Logger
from utils.seeding import set_seed

def train_teacher_ssl(config):
    set_seed(42)
    device = torch.device("cuda")

    log_dir = os.path.join(config["save_dir"],"tensorboard")
    logger = Logger(log_dir)

    train_loader = get_ssl_dataloaders(
        dataset_name=config["dataset"],
        batch_size=config["batch_size"],
        aug_type="barlow_twins",
        data_root=config["data_path"],
        train=True
    )

    model = build_model('resnet18').to(device)
    loss_fn = BarlowTwins(lambda_bt=config["lambda_bt"])

    optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config["epochs"])

    all_trainable = all(p.requires_grad for p in model.parameters())
    print("All parameters require grad:", all_trainable)
    
    model.train()
    for epoch in range(config["epochs"]):
        total_loss = 0.0

        for (x1,x2), _ in tqdm(train_loader,desc=f"Epoch {epoch+1}"):
            x1,x2 = x1.to(device),x2.to(device)

            z1 = model(x1,project=True)
            z2 = model(x2,project=True)

            loss = loss_fn(z1,z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss/len(train_loader)
        logger.log(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}", step=epoch, tag="ssl/loss")
        logger.log(f"{scheduler.get_last_lr()[0]:.6f}", step=epoch, tag="ssl/lr")


        scheduler.step()

        ckpt_path = os.path.join(config["save_dir"], f"teacher_epoch_{epoch+1}.pth")
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, path=ckpt_path)

    logger.close()
    print("Training complete.")


config = {
    "dataset": "cifar100",
    "data_path": "./data",
    "batch_size": 512,
    "lambda_bt": 5e-3,
    "lr": 0.01,
    "weight_decay": 1e-6,
    "epochs": 300,
    "save_dir": "./checkpoints/teacher_ssl",
    "seed": 42
}

train_teacher_ssl(config)