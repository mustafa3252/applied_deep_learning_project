# Import and expose dataset classes for easier imports elsewhere
# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
# Gen AI used: Claude,Chatgpt and Co-Pilot
# Extra library Usage: We used more than three pip installs but this code was for the experiment 
# and these extra libraries are not used for the final implementation of weakly supervised segmentation framework
from .oxford_pet import OxfordPetClassification, OxfordPetSegmentation, PseudoLabeledDataset

__all__ = [
    'OxfordPetClassification',
    'OxfordPetSegmentation',
    'PseudoLabeledDataset'
]