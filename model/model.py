import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm


class efficientnet_b0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0(pretrained=True)
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return self.model(x)

class efficientnet_b3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.efficientnet_b3(pretrained=True)
        self.model.classifier[1] = nn.Linear(in_features=1536, out_features=num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.model.classifier[1].weight)
        stdv = 1. / math.sqrt(self.model.classifier[1].weight.size(1))
        self.model.classifier[1].bias.data.uniform_(-stdv, stdv)
        # self.model.classifier[2] = nn.SiLU(inplace=True)
        # self.model.classifier[3] = nn.Dropout(p=0.5, inplace=True)
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return self.model(x)

class VisionTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

class SwinTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('swin_base_patch4_window12_384', pretrained = True, num_classes = num_classes)
    def forward(self, x):
        return self.model(x)

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return self.model(x)