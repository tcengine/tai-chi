__all__ = ["Accuracy", "BiAccuracy", "F1Score", "MeanAbsoluteError"]

import torch
from typing import Dict

class Metric:
    def __init__(self):
        pass

    def __call__(self, y_, y):
        raise NotImplementedError(
            f"{self.__class__.__name__} is not defined properly")

class Accuracy(Metric):
    """
    Accuracy for multi-class classification
    """
    def __call__(self, y_, y):
        return (y_.argmax(-1) == y).float().mean()

def recall(y_pred, y_true, bm=0.5):
    is_right = (((y_pred > bm)*1).long() == y_true.long())
    is_right_in_targ = is_right[y_true == 1]
    if len(is_right_in_targ) == 0:
        is_right_in_targ = torch.zeros(1).to(y_pred.device)
    return is_right_in_targ.float().mean()


def precision(y_pred, y_true, bm=0.5):
    is_right = (((y_pred > bm)*1).long() == y_true.long())
    is_right_in_pred = is_right[(y_pred > bm)]
    if len(is_right_in_pred) == 0:
        is_right_in_pred = torch.zeros(1).to(y_pred.device)
    return is_right_in_pred.float().mean()

class BiAccuracy(Metric):
    def __call__(self, y_, y) -> Dict[str, float]:
        return dict(acc=((y_ > .5).float() == y).float().mean())

class F1Score(Metric):
    """
    F1 for binary classification
    """
    def __call__(self, y_, y) -> Dict[str, float]:
        rec = recall(y_, y)
        prec = precision(y_, y)
        denomenator = (rec + prec)
        if denomenator <= 0.:
            f1 = denomenator
        else:
            f1 = (2*rec*prec)/denomenator
        return dict(f1=f1, rec=rec, prec=prec)

class MeanAbsoluteError(Metric):
    """
    Mean Absolute Error
    """
    def __call__(self, y_, y):
        return dict(mae=torch.abs(y_ - y).mean())

    
