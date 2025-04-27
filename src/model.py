import torch
from torch import nn
from torch.nn import functional as F
from src.vision_transformer import VisionTransformer, vit_b_16

sigmoid = nn.Sigmoid()

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

class Model(nn.Module):
    def __init__(
            self,
            image_size: int=224,
            n_meta_features: int=0,
            n_meta_dim: int=[512, 128],
            num_classes: int=1,
            droupouts: int=3,
            pretrained: bool=False,
        ):
        super(Model, self).__init__()
        self.droupouts = droupouts
        self.vit = vit_b_16(
            image_size=image_size,
            num_classes=num_classes,
            pretrained=pretrained,
        )

        in_channels = self.vit.hidden_dim

        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(0.1),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module()
            )
            in_channels += n_meta_dim[1]
        
        self.fc = nn.Sequential(
            nn.dropout(0.5),
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            Swish_Module(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            Swish_Module(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )


    def forward(
            self, 
            x: torch.Tensor,
            x_meta: torch.Tensor=None,
            ) -> torch.Tensor:
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        out = self.fc(x)
        for _ in self.dropouts - 1:
            out += self.fc(x)
        out /= len(self.dropouts)
        return out