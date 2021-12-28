__all__ = ["Enrich", "EnrichCleanTyping"]

from pathlib import Path
import json
from tai_chi_engine.utils import clean_name
from tai_chi_engine.stateful import Stateful
from typing import Union
from tai_chi_tuna.front.typer import STR, LIST


class Enrich(Stateful):
    phase_state = 'enrich'
    """
    Enrich Base Class
    Some default attributes
    - is_enrich = True
    - typing = None # output typing
    - multi_cols = False # use multi-column as input
    - prefer = None
    - lazy = False  # shall we execute enrichment only through the iteration
    - src = None # source column
    """
    is_enrich = True
    typing = None  # output typing
    multi_cols = False  # use multi-column as input
    prefer = None
    lazy = False  # shall we execute enrichment only through the iteration
    src = None  # source column

    # which properties to save to disk
    stateful_conf = dict()

    def __init__(self): pass

    def __call__(self, row):
        return row

    def rowing(self, row):
        if self.multi_cols:
            return self(row)
        else:
            return self(row[self.src])

class EnrichCleanTyping(Enrich):
    """
    Typing a column, and fill in the missing value
    The src and the dst can be same column for this case
    """
    stateful_conf = dict(to_type='str', missing='str')
    typing = Union[str, int, float, bool] # output typing
    def __init__(
        self,
        to_type: LIST(options=['String','Float','Integer','Bool'], default="Float")='Float',
        missing: STR(default="0") = "0",
        ):
        self.to_type = to_type
        self.missing = missing

        self.the_type = dict({'String':str,'Float':float,'Integer':int,'Bool':bool})[to_type]
        self.mst = self.the_type(missing)

    def __repr__(self,):
        return f"[CleanTyping:{self.to_type}, with missing value to: {self.missing}]"

    def __call__(self, x):
        if x is None:
            # return the default missing value
            return self.mst
        # return the typed x
        return self.the_type(x)
