import pandas as pd

from tai_chi_engine.utils import df_creator_image_folder
from pathlib import Path
from tai_chi_engine import TaiChiEngine

from tai_chi_tuna.config import PhaseConfig
from tai_chi_tuna.flow.to_enrich import set_enrich, execute_enrich
from tai_chi_tuna.flow.to_quantify import (
    execute_quantify, TaiChiDataset, choose_xy,
    save_qdict, load_qdict
)
from tai_chi_tuna.flow.to_model import TaiChiDataModule, assemble_model
from tai_chi_tuna.flow.to_train import (
    make_slug_name, set_trainer, run_training)

TEST_DIR = Path(__file__).parent.parent
DATA = TEST_DIR/"data"/"bear_dataset"


def end_to_end_run(base_df: pd.DataFrame, phase: PhaseConfig):
    base_df = execute_enrich(
        base_df, phase, enrichments=TaiChiEngine.enrichments_map)
    ds = TaiChiDataset(base_df)

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
    # phase.save()
    run_training(phase, final_model, datamodule)(dict(max_epochs=1))



def test_image_classification(mocker):
    # avoid download pretrained model for test
    from torchvision.models import resnet18
    model = resnet18(pretrained=False)
    mocker.patch('torchvision.models.resnet18',
                    return_value=model)
    base_df = df_creator_image_folder(DATA)

    PROJECT = TEST_DIR/"phases"/"image_problem"

    phase = PhaseConfig.load(PROJECT)

    end_to_end_run(base_df=base_df, phase=phase)

    
