import torch
from models.resnet import Reduced_ResNet18, SupConResNet, Reduced_ResNet18_DVC
from models.pretrained_resnet import Reduced_ResNet18_pre, Reduced_ResNet18_DVC_pre
from models.resnet_others import resnet32
from torchvision import transforms
from models.resnet_others import SupConResNet_res32
import torch.nn as nn


default_trick = {'labels_trick': False, 'kd_trick': False, 'separated_softmax': False,
                 'review_trick': False, 'ncm_trick': False, 'kd_trick_star': False}


input_size_match = {
    'cifar100': [3, 32, 32],
    'cifar10': [3, 32, 32],
    'core50': [3, 128, 128],
    'mini_imagenet': [3, 84, 84],
    'openloris': [3, 50, 50],
    'vfn': [3, 224, 224],
    'imagenet_subset': [3,224,224]
}


n_classes = {
    'cifar100': 100,
    'cifar10': 10,
    'core50': 50,
    'mini_imagenet': 100,
    'openloris': 69,
    'vfn': 74,
    'imagenet_subset':100
}


transforms_match = {
    'core50': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'cifar100': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(4),
        transforms.RandomResizedCrop(32),
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4866, 0.4409], [0.2009, 0.1984, 0.2023]),
        ]),
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'mini_imagenet': transforms.Compose([
        transforms.ToTensor()]),
    'openloris': transforms.Compose([
            transforms.ToTensor()]),
    'vfn': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        #transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
    'imagenet_subset': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        #transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4866, 0.4409], [0.2009, 0.1984, 0.2023])
    ])
}

class_distribution_table = {
                            'lt': [21, 36, 13, 344, 171, 10, 7, 24, 15, 14, 77, 7, 434, 6, 38, 328, 149, 12, 67, 85, 33, 19, 13, 477, 
                            9, 206, 226, 48, 135, 42, 273, 11, 61, 11, 378, 32, 10, 237, 248, 64, 7, 74, 17, 30, 12, 44, 197, 
                            314, 118, 40, 89, 6, 260, 18, 5, 5, 5, 455, 25, 23, 70, 179, 98, 9, 163, 102, 8, 188, 5, 500, 8, 
                            142, 216, 6, 299, 286, 56, 156, 123, 58, 27, 20, 93, 29, 361, 26, 15, 396, 112, 415, 46, 53, 16, 
                            6, 81, 22, 129, 51, 35, 107],

                            'ltio': [500, 477, 455, 434, 415, 396, 378, 361, 344, 328, 314, 299, 286, 273, 260, 248, 237, 226, 
                            216, 206, 197, 188, 179, 171, 163, 156, 149, 142, 135, 129, 123, 118, 112, 107, 102, 98, 93, 89, 85, 
                            81, 77, 74, 70, 67, 64, 61, 58, 56, 53, 51, 48, 46, 44, 42, 40, 38, 36, 35, 33, 32, 30, 29, 27, 26, 
                            25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 15, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 
                            7, 7, 6, 6, 6, 6, 5, 5, 5, 5],

                            'conv': [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 
                            500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 
                            500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 
                            500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 
                            500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500]
}

class_distribution_table_imagenet = {
                            'lt': [54, 96, 34, 896, 445, 26, 19, 63, 41, 37, 202, 20, 1130, 18, 100, 855, 387, 31, 175, 221, 87, 50, 
                            36, 1240, 24, 537, 589, 127, 353, 110, 710, 30, 160, 28, 983, 83, 27, 617, 647, 167, 18, 193, 45, 79, 32, 
                            115, 512, 816, 307, 105, 232, 15, 677, 47, 14, 13, 14, 1184, 66, 60, 184, 467, 255, 23, 425, 267, 22, 489, 
                            13, 1300, 21, 370, 562, 16, 779, 743, 146, 406, 322, 152, 72, 52, 243, 76, 938, 69, 39, 1030, 293, 1079, 
                            121, 139, 43, 17, 211, 57, 337, 133, 91, 280],

                            'ltio': [1300, 1240, 1184, 1130, 1079, 1030, 983, 938, 896, 855, 816, 779, 743, 710, 677, 647, 617, 589, 
                            562, 537, 512, 489, 467, 445, 425, 406, 387, 370, 353, 337, 322, 307, 293, 280, 267, 255, 243, 232, 221, 
                            211, 202, 193, 184, 175, 167, 160, 152, 146, 139, 133, 127, 121, 115, 110, 105, 100, 96, 91, 87, 83, 79, 
                            76, 72, 69, 66, 63, 60, 57, 54, 52, 50, 47, 45, 43, 41, 39, 37, 36, 34, 32, 31, 30, 28, 27, 26, 24, 23, 22, 
                            21, 20, 19, 18, 18, 17, 16, 15, 14, 14, 13, 13],

                            'conv': [1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 
                            1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 
                            1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 
                            1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 
                            1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 
                            1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300]
}
                            

class_order = {
    'cifar100_conv': [
            68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
        ],
    'cifar100_lt': [
            68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
        ],
    'cifar100_ltio': [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 
            57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
             84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99
        ],
    'imagenet_subset_conv': [
            68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
        ],
    'imagenet_subset_lt': [
            68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
        ],
    'imagenet_subset_ltio': [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 
            57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
             84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99
        ],
    'vfn_lt': [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 
            57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73
    ],
    'vfn_ltio': [
        70, 28, 63, 20, 33, 57, 7, 64, 51, 65, 1, 40, 68, 73, 43, 12, 58, 71, 18, 15, 21, 19, 45, 49, 29, 60, 4, 42, 69, 31, 
        3, 37, 25, 48, 50, 52, 62, 36, 11, 32, 16, 56, 35, 26, 10, 46, 39, 44, 53, 5, 14, 24, 72, 55, 23, 66, 6, 0, 67, 22, 
        13, 54, 30, 41, 47, 27, 8, 59, 34, 61, 2, 17, 9, 38]
}


def setup_architecture(params):
    nclass = n_classes[params.data]
    if params.agent in ['SCR', 'SCP']:
        if params.data == 'mini_imagenet':
            return SupConResNet(640, head=params.head)
        elif params.data == 'cifar100':
            return SupConResNet_res32(nclasses = nclass, dim_in=64, head=params.head)
        elif params.data == 'vfn' or params.data == 'imagenet_subset':
            return SupConResNet_pre(nclasses = nclass, dim_in=512, head=params.head)
        else:
            return SupConResNet_pre(nclasses = nclass, head=params.head)
    if params.agent == 'CNDPM':
        from models.ndpm.ndpm import Ndpm
        return Ndpm(params)
    if params.data == 'cifar100':
        if params.agent == 'ER_DVC':
            return Reduced_ResNet18_DVC(nclass)
        else:
            return Reduced_ResNet18(nclass)
    elif params.data == 'vfn':
        if params.agent == 'ER_DVC':
            # model =  resnet18_pre_DVC(nclass, pretrained=False)
            model = Reduced_ResNet18_DVC_pre(nclass)
            model.backbone.linear = nn.Linear(160, nclass, bias=True)
        else:
            model = Reduced_ResNet18_pre(nclass)
            model.backbone.linear = nn.Linear(160, nclass, bias=True)
        return model
    elif params.data == 'imagenet_subset':
        if params.agent == 'ER_DVC':
            model =  Reduced_ResNet18_DVC_pre(nclass)
            model.backbone.linear = nn.Linear(160, nclass, bias=True)
        else:
            model = Reduced_ResNet18(nclass)
            model.backbone.linear = nn.Linear(640, nclass, bias=True)
        return model


def setup_opt(optimizer, model, lr, wd):
    if optimizer == 'SGD':
        optim = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                weight_decay=wd)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=wd)
    else:
        raise Exception('wrong optimizer name')
    return optim
