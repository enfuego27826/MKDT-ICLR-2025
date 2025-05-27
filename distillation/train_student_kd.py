import torch
import torch.nn as nn
from tqdm import tqdm

from dataloader.dataloader import get_ssl_dataloaders
from models.builder import build_model
from losses.kd_loss import KDLoss
from utils.checkpoints import save_checkpoint
from utils.seeding import set_seed

def train_student_kd(config):
    set_seed(42)
    device = torch.device("cuda")

    teacher = build_model("resnet18").to(device)
    teacher.load_state_dict(torch.load(config["teacher_ckpt"]))
    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad = False

    train_loader = get_ssl_dataloader(
        dataset_name=config["dataset"],
        batch_size=config["batch_size"],
        aug_type="none",
        data_root=config["data_path"],
        train = True
    )

    kd_loss = KDLoss()

    for sid in range(config["num_students"]):
        print(f"Training student {sid+1}/{config["num_students"]}")
        student = build_model("convnet").to(device)
        optimizer = torch.optim.SGD(student.parameters(),lr=config["lr"],momentum=0.9,weight_decay=1e-4)

        for epoch in range(config["epochs"]):
            student.train()
            total_loss = 0.0

            for images,_ in tqdm(train_loader,desc=f"Student {sid} | Epoch {epoch+1}"):
                images = images.to(device)

                with torch.no_grad():
                    z_teacher = teacher.get_representation(images)
                
                z_student = student.get_representation(images)
                loss = kd_loss(z_student,z_teacher)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss/len(train_loader)

            print(f"[Student {sid}] Epoch {epoch+1}: KD Loss = {avg_loss:.4f}")

            save_path = os.path.join(config["save_dir"],f"student_{sid+1}",f"epoch_{epoch+1}.pth")
            os.makedirs(os.path.dirname(save_path),exist_ok=True)
            save_checkpoint({
                "epoch": epoch+1,
                "model_state_dict": student.state_dict()
            }, save_path)

config = {
    "dataset": "cifar100",
    "data_path": "./data",
    "teacher_ckpt": "./checkpoints/teacher_ssl/teacher_epoch_282.pth",
    "save_dir": "./trajectories",
    "batch_size": 128,
    "lr": 0.1,
    "epochs": 20,
    "num_students": 100,
    "seed": 42
}