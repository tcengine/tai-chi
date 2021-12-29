__all__ = ["ImageConvEncoder"]

from tai_chi_tuna.front.typer import LIST
from tai_chi_tuna.flow.to_quantify import BATCH_SIZE

from .basic import EntryModel, Empty

from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    resnext101_32x8d, resnext50_32x4d)

RESNET_OPTIONS = {"resnet18": resnet18,
                  "resnet34": resnet34,
                  "resnet50": resnet50,
                  "resnet101": resnet101,
                  "resnet152": resnet152,
                  "resnext101_32x8d": resnext101_32x8d,
                  "resnext50_32x4d": resnext50_32x4d}

class ImageConvEncoder(EntryModel):
    def __init__(self, model):
        super().__init__()
        self.name = "cnn"
        self.output_shape = (BATCH_SIZE, model.fc.in_features)
        self.out_features = model.fc.in_features
        model.fc = Empty()
        self.model = model

    def forward(self, data):
        return self.model(data)

    def __repr__(self):
        return f"""ComputerVisionEncoder: {self.name}
        Outputs shape:{self.output_shape}"""

    @classmethod
    def from_quantify(
        cls,
        quantify,
        name: LIST(options=list(
            RESNET_OPTIONS.keys()), default="resnet18"),
    ):
        # do not load pretrained for inference
        if quantify.is_inference:
            model = RESNET_OPTIONS[name](pretrained=False)
        # load pretrained for transfer learning
        else:
            model = RESNET_OPTIONS[name](pretrained=True, progress=True,)
        obj = cls(model)
        obj.name = name
        return obj