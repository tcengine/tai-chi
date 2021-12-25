from tai_chi_engine.quantify import (
    QuantifyCategory, QuantifyMultiCategory, QuantifyNum
)
__all__ = ['ExitModel', 'RegressionTop', 'CategoryTop',
           'MultiCategoryTop', 'ALL_EXIT', 'QUANTIFY_2_EXIT_MAP']

from .category import CategoryTop, MultiCategoryTop

from .basic import ExitModel, RegressionTop

ALL_EXIT = dict(
    CategoryTop=CategoryTop,
    MultiCategoryTop=MultiCategoryTop,
    RegressionTop=RegressionTop,
)

QUANTIFY_2_EXIT_MAP = dict({
    QuantifyCategory: [
        CategoryTop,
    ],
    QuantifyMultiCategory: [
        MultiCategoryTop,
    ],
    QuantifyNum: [
        RegressionTop,
    ],
})
