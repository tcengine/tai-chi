__all__ = ["TaiChiLearn",]
from tai_chi_tuna.flow.trunk import TaiChiLearn
from .enrich import ENRICHMENTS
from .quantify import QUANTIFY
from .model import (ALL_EXIT, QUANTIFY_2_EXIT_MAP,
                    ALL_ENTRY, QUANTIFY_2_ENTRY_MAP)

TaiChiLearn.enrichments_map = ENRICHMENTS
TaiChiLearn.quantify_map = QUANTIFY
TaiChiLearn.quantify_2_entry_map = QUANTIFY_2_ENTRY_MAP
TaiChiLearn.quantify_2_exit_map = QUANTIFY_2_EXIT_MAP
TaiChiLearn.all_entry = ALL_ENTRY
TaiChiLearn.all_exit = ALL_EXIT

