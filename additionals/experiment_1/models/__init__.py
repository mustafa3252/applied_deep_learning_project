# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
 
# Gen AI used: Claude, ChatGPT and Co-Pilot

# For the changes in this specific experiment compared to the main code, Gen AI was not used



from .classifier import get_classifier
from .segmentation import get_segmentation_model
from .dropout import resnet_insert_dropout, fcn_insert_dropout

__all__ = [
    'get_classifier',
    'get_segmentation_model',
    "resnet_insert_dropout",
    "fcn_insert_dropout"
]