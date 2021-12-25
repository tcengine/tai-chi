__all__ = ["CategoryTop", "MultiCategoryTop"]

from .metric import accuracy, bi_accuracy
from .loss import BCEWithLogitsLossCasted
from .basic import ExitModel, nn


class CategoryTop(ExitModel):
    prefer = "CrossEntropyLoss"
    input_dim = 2

    def __init__(self, in_features, out_features):
        super().__init__()
        self.top = nn.Linear(
            in_features=in_features, out_features=out_features)
        self.activation = nn.Softmax(dim=-1)
        self.crit = nn.CrossEntropyLoss()
        self.metric_funcs.update({"acc": accuracy})

    def forward(self, x):
        return self.top(x)

    @classmethod
    def from_quantify(cls, quantify, entry_part):
        out_features = len(quantify.category)
        in_features = entry_part.out_features
        return cls(
            in_features=in_features,
            out_features=out_features,
        )


class MultiCategoryTop(ExitModel):
    prefer = "BCEWithLogitsLossCasted"
    input_dim = 2

    def __init__(self, in_features, out_features):
        super().__init__()
        self.top = nn.Linear(
            in_features=in_features, out_features=out_features)
        self.activation = nn.Sigmoid()
        self.crit = BCEWithLogitsLossCasted()
        self.metric_funcs.update({"acc": bi_accuracy})

    def forward(self, x):
        return self.top(x)

    @classmethod
    def from_quantify(cls, quantify, entry_part):
        out_features = len(quantify.category)
        in_features = entry_part.out_features
        return cls(
            in_features=in_features,
            out_features=out_features,
        )
