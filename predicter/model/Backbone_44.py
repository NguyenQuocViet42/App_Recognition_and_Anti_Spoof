from torchvision import models
import torch

class Backbone_44(torch.nn.Module):
    def __init__(self):
        super(Backbone_44, self).__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet50 = models.resnet50(pretrained = 'ResNet50_Weights.DEFAULT')
        resnet50.to(device)
        # for param in resnet50.parameters():
        #     param.requires_grad = False
        
        # count = 0
        # for param in resnet50.parameters():
        #     if count == 100:
        #         break
        #     param.requires_grad = False
        #     count += 1
        
        selected_layers = list(resnet50.children())[:-2]
        selected_layers[-1] = selected_layers[-1][0]
        
        self.backbone = torch.nn.Sequential(*selected_layers)

    def forward(self, x):
        x = self.backbone(x)
        return x