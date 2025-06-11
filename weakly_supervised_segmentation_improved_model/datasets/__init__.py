# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
# Gen AI used: Claude,Chatgpt and Co-Pilot

"""
Expose dataset classes for classification, segmentation, and pseudo-labeling tasks.
"""

from .oxford_pet import OxfordPetClassification, OxfordPetSegmentation, PseudoLabeledDataset

__all__ = [
    'OxfordPetClassification',
    'OxfordPetSegmentation',
    'PseudoLabeledDataset'
]