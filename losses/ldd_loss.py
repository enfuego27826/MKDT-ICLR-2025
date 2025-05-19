import torch
import torch.nn.functional as flatten

def ldd_loss(theta_student,theta_expert_start,theta_expert_target,eps=1e-8):
    """
    MTT Loss (From MKDT Paper)

    Args:
        theta_student: flattened weights from synthetic student model
        theta_expert_start: starting flattened weights from expert model
        theta_expert_target: target to be reached by student model
        eps: small constant to avoid division by zero

    Returns:
    MTT Loss
    """

    num = F.mse_loss(theta_student,theta_expert_target,reduction='sum')
    den = F.mse_loss(theta_expert_start,theta_expert_target,reduction='sum') + eps
    loss = num/den

    return loss

