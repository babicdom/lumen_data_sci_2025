import torch
from torch import nn
from torch.nn import functional as F
from src.vision_transformer import VisionTransformer, vit_b_16

class Model(nn.Module):
    def __init__(
            self,
            image_size=224,
            num_classes=1
        ):
        super(Model, self).__init__()
        self.vit = vit_b_16(
            image_size=image_size,
            num_classes=num_classes,
            pretrained=True
        )
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x