from tai_chi_engine.utils import df_creator_image_folder
from pathlib import Path

TEST_DIR = Path(__file__).parent.parent
DATA = TEST_DIR/"data"/"bear_dataset"

def test_image_classification():
    df = df_creator_image_folder(DATA)
    assert True
