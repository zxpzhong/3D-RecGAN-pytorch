from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import torch.nn as nn

class efficientnet(nn.Module):
    def __init__(self, num_classes=1000):
        super(efficientnet, self).__init__()
        self.encoder = EfficientNet.from_pretrained('efficientnet-b1',num_classes=num_classes) 
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(1280,num_classes)

    def forward(self, x):
        feature = self.encoder.extract_features(x)
        feature = self.GAP(feature)
        feature = feature.view(feature.size(0), -1)
        logit = self.classifier(feature)
        return logit
    def extract_feature(self,x):
        feature = self.encoder.extract_features(x)
        feature = self.GAP(feature)
        feature = feature.view(feature.size(0), -1)
        return feature