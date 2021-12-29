__all__ = ["ExitModel", "RegressionTop", "HIDDEN_SIZE_OPTIONS"]

from torch import nn
import numpy as np
from .metric import MeanAbsoluteError
from tai_chi_tuna.front.typer import LIST


class ExitModel(nn.Module):
    """
    # Base Model of exit part

    ## You **have to** set the following for the inherited class
    - self.crit: the criterion, a callble to calc ```self.crit(y_, y)```
    - self.from_quantify: a class method to build the model from entry_dict and quantify class

    ## You **can** set/ overwrite the following for the inherited class
    - self.metric_funcs: a dict of metric functions, key is the name of the metric
    - self.activation: the activation function, a callble to calc ```self.activation(y_)```
    - self.loss_step: a callble to calc ```self.loss_step(x, y)```
    - self.eval_forward: a callble to calc ```self.eval_forward(x)```
    """
    metric_funcs = dict()

    def loss_step(self, x, y):
        y_ = self(x)
        loss = self.crit(y_, y)
        metrics = dict()
        if hasattr(self, "activation"):
            y_ = self.activation(y_)
        # calculate metrics key by key
        for k, func in self.metric_funcs.items():
            metric = func(y_, y)
            if hasattr(metric, "keys"):
                metrics.update(metric)
            else:
                metrics[k] = metric
        return dict(loss=loss, y_=y_, **metrics)

    def create_top(
            self, in_features: int,
            out_features: int,
            hidden_size: int = 0) -> nn.Module:
        """
        Create the top part of the model
        if hidden_size is 0, then no hidden layer (only a linear layer)
        """
        if hidden_size > 0:
            self.top = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(in_features=hidden_size, out_features=out_features),
            )
        else:
            self.top = nn.Linear(
                in_features=in_features, out_features=out_features)
        return self.top

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


HIDDEN_SIZE_OPTIONS = LIST(default=0, options=[0, 64, 128, 256, 512, 1024, 2048])


class RegressionTop(ExitModel):
    prefer = "MSELoss"
    input_dim = 2

    def __init__(self, in_features, out_features, hidden_size=0):
        super().__init__()
        self.create_top(in_features, out_features, hidden_size)
        self.crit = nn.MSELoss()
        self.metric_funcs.update(
            {
                "mae": MeanAbsoluteError(),
            })

    def forward(self, x):
        return self.top(x)

    @classmethod
    def from_quantify(
        cls,
        quantify,
        entry_part,
        hidden_size: HIDDEN_SIZE_OPTIONS = 0):
        out_features = 1
        in_features = entry_part.out_features
        return cls(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
        )
