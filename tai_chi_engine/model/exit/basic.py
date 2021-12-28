__all__ = ["ExitModel", "RegressionTop"]

from torch import nn
import numpy as np


class ExitModel(nn.Module):
    metric_funcs = dict()

    def loss_step(self, x, y):
        y_ = self(x)
        loss = self.crit(y_, y)
        metrics = dict()
        if hasattr(self, "activation"):
            y_ = self.activation(y_)
        for k, func in self.metric_funcs.items():
            metrics.update({k: func(y_, y)})
        return dict(loss=loss, y_=y_, **metrics)

    def eval_forward(self, x) -> np.ndarray:
        """
        Forward pass for 
        """
        y_ = self(x)
        if hasattr(self, "activation"):
            y_ = self.activation(y_)
        return y_

    @classmethod
    def from_quantify(cls, ):
        raise ImportError(
            f"Please define class function 'from_quantify' for {cls.__name__}"
        )


class RegressionTop(ExitModel):
    prefer = "MSELoss"
    input_dim = 2

    def __init__(self, in_features, out_features):
        super().__init__()
        self.top = nn.Linear(
            in_features=in_features, out_features=out_features)
        self.crit = nn.MSELoss()

    def forward(self, x):
        return self.top(x)

    @classmethod
    def from_quantify(cls, quantify, entry_part):
        out_features = 1
        in_features = entry_part.out_features
        return cls(
            in_features=in_features,
            out_features=out_features,
        )
