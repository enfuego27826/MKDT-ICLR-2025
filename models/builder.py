from models.convnet import ConvNet
from models.resnet import ResNet18

def build_model(name,**kwargs):
    """
    Returns an instance of a model based on name

    Args:
        name (str): One of 'convnet', 'resnet18'
        **kwargs: Keyword arguments passed to model constructor

    Returns:
        nn.Module
    """

    if name == 'convnet':
        return ConvNet(**kwargs)

    elif name == 'resnet18':
        return ResNet18(**kwargs)

    else:
        raise ValueError(f"Unsupported model: {name}")