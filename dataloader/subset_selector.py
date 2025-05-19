import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

def compute_student_loss(student_model,dataloader,z_teacher):
    """ Computer per-example MSE between student and teacher representations

    Args:
        student model (nn.Module): Trained student
        dataloader (DataLoader): Dataloader for real dataset
        z_teacher (Tensor): Precomputed teacher representation

    Returns:
        Dict[int,float]: Mapping from dataset index to MSE
    """

    student_model.eval()
    loss_dict = {}

    with torch.no_grad():
        for batch_idx, (x,idxs) in enumerate(tqdm(dataloader)):
            x = x.cuda()
            idxs = idxs.cuda() if torch.is_tensor(idxs) else idxs
            student_reps = student_model.get_representation(x)
            teacher_reps = z_teacher[idxs]

            mse = F.mse_loss(student_reps,teacher_reps,reduction='none')

            for i,idx in enumerate(idxs):
                loss_dict[idx] = mse[i]

    return loss_dict

def select_high_loss_img(loss_dict,k):
    """Returns top-k indexes with highest loss.

    Args:
        loss_dict (dict): index -> loss (dictionary)
        k (int): How many images to keep

    Returns:
        List[int]: Sorted list of top k indices
    """

    sorted_items = sorted(loss_dict.items(),key = lambda x: x[1],reverse=True)
    top_indexes = [idx for idx, _ in sorted_items[:k]]

    return top_indexes

def compute_high_loss_subset(dataset,z_teacher,student_checkpoints,k,model_builder,batch_size=128):
    """Compute average loss across K students and returns top K images.

    Args:
        dataset (Dataset): Real dataset
        z_teacher (Tensor): Precomputed teacher reps
        student_checkpoints (List[str]): Paths to student weights
        k (int): Number of high-loss examples to select
        model_builder (callable): Returns a student model
        batch_size (int): Dataloader batch size
    
    Returns:
        List[int]: Indexes of top k high-loss examples
    """

    net = {}

    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=2)

    for path in student_checkpoints:
        student = model_builder.cuda()
        student.load_state_dict(torch.load(path)['model_state_dict'])

        student_loss = compute_student_loss(student,dataloader,z_teacher)

        for idx,loss in student_loss.items():
            net.setdefault(idx,[]).append(loss)
    
    avg_loss_dict = {idx: np.mean(losses) for idx,losses in net.items()}
    return select_high_loss_img(avg_loss_dict,k)