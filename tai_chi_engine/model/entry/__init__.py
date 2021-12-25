__all__ = ['EntryModel', 'Empty', 'ImageConvEncoder',
           'CategoryEncoder', 'MultiCategoryEncoder', 'TransformerEncoder',
           'ALL_ENTRY', 'QUANTIFY_2_ENTRY_MAP'
           ]

from .basic import (
    EntryModel, Empty,
)
from .image import (
    ImageConvEncoder,
)
from .category import (
    CategoryEncoder, MultiCategoryEncoder
)
from .text import (
    TransformerEncoder
)

from tai_chi_engine.quantify import (
    QuantifyImage, QuantifyText, 
    QuantifyCategory, QuantifyMultiCategory, QuantifyNum
)

ALL_ENTRY = dict(
    ImageConvEncoder=ImageConvEncoder,
    CategoryEncoder=CategoryEncoder,
    MultiCategoryEncoder=MultiCategoryEncoder,
    TransformerEncoder=TransformerEncoder,
    Empty=Empty,
)

QUANTIFY_2_ENTRY_MAP = dict({
    QuantifyImage:[
        ImageConvEncoder,
    ],
    QuantifyCategory:[
        CategoryEncoder,
    ],
    QuantifyMultiCategory:[
        MultiCategoryEncoder,
    ],
    QuantifyText:[
        TransformerEncoder,
    ],
    QuantifyNum:[
        Empty,
    ],
})