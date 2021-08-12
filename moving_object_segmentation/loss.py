import torch
import torch.nn.functional as F


# dice loss
def dice_loss(pred, target):
    pred = pred.contiguous().view(pred.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(pred * target, 1)
    b = torch.sum(pred * pred, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1 - d


# combined loss based on dice loss and binary cross entropy
def calc_loss(pred, target, bce_weight=0.5):
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    bce = F.binary_cross_entropy_with_logits(pred, target).type(dtype)

    pred = torch.sigmoid(pred).type(dtype)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    return loss
