from .ResNet import ResNet18, ResNet18Block, ResNet50Block
from .ResNetCifar import ResNet18Cifar, ResNet18CifarBlock, ResNet32CifarBlock
from .LDAMLoss import LDAMLoss


def load_net(network, classes=4):
    # Load model
    if network == 'ResNet-18':
        model = ResNet18(classes)
    elif network == 'ResNet-18-block':
        model = ResNet18Block(classes)
    elif network == 'ResNet-50-block':
        model = ResNet50Block(classes)
    elif network == 'ResNet-18-cifar':
        model = ResNet18Cifar(classes)
    elif network == 'ResNet-18-cifar-block':
        model1, model2 = ResNet18CifarBlock(classes)
        return model1, model2
    elif network == 'ResNet-32-cifar-block':
        model1, model2 = ResNet32CifarBlock(classes)
        return model1, model2
    else:
        raise NotImplementedError('Network {0} not implemented'.format(network))
    return model