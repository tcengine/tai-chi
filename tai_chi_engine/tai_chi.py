__all__ = ["TaiChiEngine",]
from tai_chi_tuna.flow.trunk import TaiChiLearn
from .enrich import ENRICHMENTS
from .quantify import QUANTIFY
from .model import (ALL_EXIT, QUANTIFY_2_EXIT_MAP,
                    ALL_ENTRY, QUANTIFY_2_ENTRY_MAP)


class TaiChiEngine(TaiChiLearn):
    """
    TaiChiEngine
    One and the only thing the end user has to initiate and operate.
    """
    
    enrichments_map = ENRICHMENTS
    quantify_map = QUANTIFY
    quantify_2_entry_map = QUANTIFY_2_ENTRY_MAP
    quantify_2_exit_map = QUANTIFY_2_EXIT_MAP
    all_entry = ALL_ENTRY
    all_exit = ALL_EXIT

