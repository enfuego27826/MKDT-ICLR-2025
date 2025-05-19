import torch
import torch.nn.functional as F

def kd_loss(student_rep,teacher_rep):
    """
    MSE between teacher and student rep.

    Args:
    student_rep : Student Represenations
    teacher_rep : Teacher Representations

    Returns:
    MSE Loss
    """

    return F.mse_loss(student_rep,teacher_rep)