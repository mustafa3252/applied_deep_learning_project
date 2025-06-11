# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
 
# Gen AI used: Claude, ChatGPT and Co-Pilot



# Import and expose dataset classes for easier imports elsewhere
from .oxford_pet import OxfordPetClassification, OxfordPetSegmentation, PseudoLabeledDataset

__all__ = [
    'OxfordPetClassification',
    'OxfordPetSegmentation',
    'PseudoLabeledDataset'
]