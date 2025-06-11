# Import and expose dataset classes for easier imports elsewhere
from .oxford_pet import OxfordPetClassification, OxfordPetSegmentation, PseudoLabeledDataset

__all__ = [
    'OxfordPetClassification',
    'OxfordPetSegmentation',
    'PseudoLabeledDataset',
]