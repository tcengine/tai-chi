__all__ = ["QUANTIFY","Quantify", "QuantifyImage", "QuantifyText",
           "QuantifyCategory", "QuantifyMultiCategory", "QuantifyNum"]

from .basic import (
    Quantify, QuantifyImage, QuantifyText, 
    QuantifyCategory, QuantifyMultiCategory, QuantifyNum
    )

QUANTIFY = dict(
    Quantify=Quantify,
    QuantifyNum=QuantifyNum,
    QuantifyImage=QuantifyImage,
    QuantifyCategory=QuantifyCategory,
    QuantifyMultiCategory=QuantifyMultiCategory,
    QuantifyText=QuantifyText,
)