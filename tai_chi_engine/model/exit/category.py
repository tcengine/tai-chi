__all__ = ["CategoryTop", "MultiCategoryTop"]

from .metric import Accuracy, BiAccuracy, F1Score
from .loss import BCEWithLogitsLossCasted
from .basic import ExitModel, nn, HIDDEN_SIZE_OPTIONS


class CategoryTop(ExitModel):
    prefer = "CrossEntropyLoss"
    input_dim = 2

    def __init__(self, in_features, out_features, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.create_top(in_features, out_features, hidden_size)
        self.activation = nn.Softmax(dim=-1)
        self.crit = nn.CrossEntropyLoss()
        self.metric_funcs.update(
            {
                "acc": Accuracy(),
            })

    def forward(self, x):
        return self.top(x)

    @classmethod
    def from_quantify(
        cls,
        quantify,
        entry_part,
        hidden_size: HIDDEN_SIZE_OPTIONS = 0,
    ):
        """
        Build the exit model parts
            with the quantify and entry parts
        """
        out_features = len(quantify.category)
        in_features = entry_part.out_features
        return cls(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
        )


class MultiCategoryTop(ExitModel):
    prefer = "BCEWithLogitsLossCasted"
    input_dim = 2

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: int = 0
    ):
        super().__init__()
        self.create_top(in_features, out_features, hidden_size)
        self.activation = nn.Sigmoid()
        self.crit = BCEWithLogitsLossCasted()
        self.metric_funcs.update(
            {"acc": BiAccuracy(), "F1": F1Score()})

    def forward(self, x):
        return self.top(x)

    @classmethod
    def from_quantify(
            cls,
            quantify,
            entry_part,
            hidden_size: HIDDEN_SIZE_OPTIONS = 0,):
        """
        Build the exit model parts
            with the quantify and entry parts
        """
        out_features = len(quantify.category)
        in_features = entry_part.out_features
        return cls(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
        )
