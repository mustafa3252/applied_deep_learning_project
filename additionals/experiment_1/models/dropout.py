# Gen AI Usage Statement:
# Gen AI was not used in this specific file



from torch.nn import Sequential, Dropout2d
from torchvision.models import ResNet
from torchvision.models.segmentation import FCN
def seq_insert_dropout(layer: Sequential, prob: float | list[float], rescale: bool):
    num_layers = len(layer)
    if isinstance(prob, float):
        prob = [prob] * num_layers

    if rescale:
        exp = 1 / num_layers
        prob = [p ** exp for p in prob]

    zipped = list(zip(layer, [Dropout2d(p) for p in prob]))
    return Sequential(*[item for pair in zipped for item in pair])

def seq_append_dropout(layer: Sequential, prob: float, rescale: bool):
    return Sequential(*[*layer, Dropout2d(
        prob if rescale else prob ** len(layer)
        )])

def resnet_insert_dropout(model: ResNet, prob: float | tuple[float, float, float, float],
                          mode: int=1, rescale: bool=True, device=None):
    """
    IN-PLACE!!!\n
    Inserts Dropout2d layers in your ResNet\n
    mode = 1 -> Inserts after each Bottleneck layer within each Sequential layer in the model\n
    mode = 2 -> Appends to each Sequential layer in the model\n
    For rescale, it applies to both modes\n
    rescale=False and mode=2 -> will self-multiply probabilities according to length of each Sequential in the model

    Args:
        model: ResNet - model in question
        prob: float or tuple[float] - probabilities to use, tuple size = 4
        mode: int - (default) 1, (read above)
        rescale: bool - (default) True, whether to scale (p ** (1 / len(sequential))) probabilities for each Sequential
    """
    if isinstance(prob, float):
        prob = [prob] * 4
    else: # Assume tuple
        assert len(prob) == 4, "Use a tuple of size 4!"

    assert mode < 3, "Invalid mode"

    if mode == 1:
        model.layer1 = seq_insert_dropout(model.layer1, prob[0], rescale=rescale)
        model.layer2 = seq_insert_dropout(model.layer2, prob[1], rescale=rescale)
        model.layer3 = seq_insert_dropout(model.layer3, prob[2], rescale=rescale)
        model.layer4 = seq_insert_dropout(model.layer4, prob[3], rescale=rescale)

    elif mode == 2:
        model.layer1 = seq_append_dropout(model.layer1, prob[0], rescale=rescale)
        model.layer2 = seq_append_dropout(model.layer2, prob[1], rescale=rescale)
        model.layer3 = seq_append_dropout(model.layer3, prob[2], rescale=rescale)
        model.layer4 = seq_append_dropout(model.layer4, prob[3], rescale=rescale)

    return model.to(device=device)

def fcn_insert_dropout(model: FCN, prob: float | tuple[float, float, float, float],
                       mode: int=1, rescale: bool=True, device=None):
    """
    IN-PLACE!!!\n
    Inserts Dropout2d layers in your FCN model's (ResNet) backbone model\n
    mode = 1 -> Inserts after each Bottleneck layer within each Sequential layer in the backbone\n
    mode = 2 -> Appends to each Sequential layer in the backbone\n
    For rescale, it applies to both modes\n
    rescale=False and mode=2 -> will self-multiply probabilities according to length of each Sequential in the backbone

    Args:
        model: FCN - model in question
        prob: float or tuple[float] - probabilities to use, tuple size = 4
        mode: int - (default) 1, (read above)
        rescale: bool - (default) True, whether to scale (p ** (1 / len(sequential))) probabilities for each Sequential
    """
    resnet_insert_dropout(model.backbone, prob, mode, rescale, device)
    return model