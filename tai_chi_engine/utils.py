from tai_chi_tuna.utils import clean_name
from pathlib import Path
import pandas as pd

def df_creator_image_folder(path: Path) -> pd.DataFrame:
    """
    Create a dataframe ,
    Which list all the image path under a system folder
    """
    path = Path(path)
    files = []
    formats = ["jpg", "jpeg", "png"]
    for fmt in formats:
        files.extend(path.rglob(f"*.{fmt.lower()}"))
        files.extend(path.rglob(f"*.{fmt.upper()}"))
    return pd.DataFrame({"path": files}).sample(frac=1.).reset_index(drop=True)