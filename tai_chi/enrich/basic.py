__all__ = ["Enrich"]


class Enrich:
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

    def __init__(self): pass

    def __call__(self, row):
        return row

    def rowing(self, row):
        if self.multi_cols:
            return self(row)
        else:
            return self(row[self.src])
