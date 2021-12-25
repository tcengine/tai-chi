from torch import nn
__all__ = ["BCEWithLogitsLossCasted", ]


class BCEWithLogitsLossCasted(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_ = nn.BCEWithLogitsLoss()

    def forward(self, y_, y):
        return self.bce_(y_, y.float())
