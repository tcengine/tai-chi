__all__ = ["end_to_end_run"]
import pandas as pd
from tai_chi_engine import TaiChiEngine

from tai_chi_tuna.config import PhaseConfig
from tai_chi_tuna.flow.to_enrich import execute_enrich
from tai_chi_tuna.flow.to_quantify import (
    execute_quantify, TaiChiDataset, 
    save_qdict
)
from tai_chi_tuna.flow.to_model import TaiChiDataModule, assemble_model
from tai_chi_tuna.flow.to_train import (
    make_slug_name, run_training)


def end_to_end_run(base_df: pd.DataFrame, phase: PhaseConfig):
    """
    Run end to end flow
    """
    # enrich
    base_df = execute_enrich(
        base_df, phase, enrichments=TaiChiEngine.enrichments_map)
    ds = TaiChiDataset(base_df)
    # quantify
    qdict = execute_quantify(df=base_df, phase=phase,
                             quantify_map=TaiChiEngine.quantify_map)
    # save quantify objects
    _ = save_qdict(phase.project, qdict)

    datamodule = TaiChiDataModule(ds, qdict)
    datamodule.configure(**phase['batch_level'])
    # assemble model
    module_zoo = {"all_entry": TaiChiEngine.all_entry,
                  "all_exit": TaiChiEngine.all_exit}

    final_model = assemble_model(phase, qdict, module_zoo)
    phase['task_slug'] = make_slug_name(phase)

    # training
    run_training(phase, final_model, datamodule)(dict(max_epochs=1))