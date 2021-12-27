__all__ = ["Enrich"]

from pathlib import Path
import json
from tai_chi_engine.utils import clean_name
from tai_chi_engine.stateful import Stateful


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
