# Gen Al Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging refactoring and refining the architecture in depth
# Gen AI used: Claude, ChatGPT and Co-Pilot

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class PetClassifier(nn.Module):
    def __init__(self, encoder=None, num_classes=37):
        super(PetClassifier, self).__init__()
        
        if encoder is None:
            encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
            encoder.fc = nn.Identity()
        
        self.encoder = encoder
        self.classifier = nn.Linear(2048, num_classes) 

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits

def get_classifier(device=None, encoder=None, num_classes=37):
    model = PetClassifier(encoder=encoder, num_classes=num_classes)
    return model.to(device) if device else model
