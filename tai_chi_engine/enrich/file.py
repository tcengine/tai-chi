__all__ = ["EnrichImage", "ParentAsLabel"]

from .basic import Enrich
from tai_chi_tuna.front.typer import STR, LIST
from pathlib import Path
from PIL import Image
import json

class EnrichImage(Enrich):
    """
    Create Image column from image path column
    """
    prefer = "QuantifyImage"
    typing = Image
    lazy = True
    stateful_dict = dict(
        convert="str",
        size="list",
    )
    def __init__(
        self, convert: STR("RGB") = "RGB",
        size: LIST(options=[28, 128, 224, 256, 512], default=224) = 224,
    ):
        self.convert = convert
        self.size = size

    def __repr__(self):
        return f"[Image:{self.size}]"

    def __call__(self, x):
        img = Image.open(x).convert(self.convert)
        img = img.resize((self.size, self.size))
        return img


class ParentAsLabel(Enrich):
    typing = str
    prefer = "QuantifyCategory"
    def __call__(self, path: Path,) -> str:
        """
        Use parent folder name as label
        """
        return Path(path).parent.name