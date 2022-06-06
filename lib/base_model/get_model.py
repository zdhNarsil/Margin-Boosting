import os, sys
import pdb
file_path = os.path.realpath(__file__)
par_path = os.path.dirname(file_path)
sys.path.append(par_path)


from cifar_resnet18 import ResNet18, ResNet152, ResNet18Ensemble
from convnet import ConvNet
from wide_resnet import WideResNet
from mnist_cnn import MnistModel


def get_model(model_name, num_classes=10):
    if model_name in ["res18"]:
        net = ResNet18(num_classes=num_classes)
    elif model_name in ["cnn"]:
        net = ConvNet()
    elif model_name in ["wres28"]:
        net = WideResNet(depth=28, num_classes=num_classes)
    elif model_name in ["wres34"]:
        net = WideResNet(depth=34, num_classes=num_classes)
    elif model_name in ["mnist"]:
        net = MnistModel()
    elif model_name in ["resens5"]:
        net = ResNet18Ensemble(ensemble_num=5, num_classes=num_classes)
    elif model_name in ["res152"]:
        net = ResNet152(num_classes=num_classes)
    else:
        raise NotImplementedError
    return net