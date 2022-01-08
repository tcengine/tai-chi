import pandas as pd

from tai_chi_engine.utils import df_creator_image_folder
from pathlib import Path

from tai_chi_tuna.config import PhaseConfig
from .run import end_to_end_run

TEST_DIR = Path(__file__).parent.parent
DATA = TEST_DIR/"data"/"bear_dataset"


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
