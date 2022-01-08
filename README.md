# Tai-Chi Engine

[![PyPI version](https://img.shields.io/pypi/v/tai-chi-engine)](https://pypi.org/project/tai-chi-engine/)
![Python version](https://img.shields.io/pypi/pyversions/tai-chi-engine)
![License](https://img.shields.io/github/license/unpackai/tai-chi)
![PyPI Downloads](https://img.shields.io/pypi/dm/tai-chi-engine)

> Powerful deep learning for civilians

> ```太极引擎```  深度学习： 强大、多模态、灵活、平民化

## Essence of the Tai-Chi Engine
* Close to state-of-the-art Deep learning, friendly to office folks, all coding-free, clicks away.
* Flexible, supporting multiple kinds of data (image, text, category, multi-category, etc), multiple x at the same time.
* Which columns to be x? Which column to be y? You can decide and play, see if the AI finds out how to guess.

## Our big pitch
* If you're a **coding muggle** - play with the engine, you'll understand ideas around Deep Learning, and have good model.
* If you're a **pro** - it's still wildly fun to try new models in 2 minutes' click, especially you have around a dozen columns.

## Playing Tai-Chi
First, tell me you are already in a jupyter notebook environment.

Open up a table, a pandas dataframe, from excel or from csv file or from SQL database, dosn't matter.
```python
import pandas as pd
df = pd.read_csv('your_data.csv')
```

Then, you can use the following code to play with the engine.
```python
from tai_chi_engine import TaiChiEngine
# load the engine
engine = TaiChiEngine(df, project="./where/to_save/your_model")
# start the playing
engine()
```

Good to go!

## Installation
Default installation:

```bash
pip install tai-chi-engine
```

## Run App on Trained Model
You can build prototype App (Based on [streamlit](https://streamlit.io/)) based on trained project, see [app](https://github.com/unpackAI/tai-chi/tree/main/tai_chi_engine/apps) part of this library.

ALL kinds of projects, only **one way** and only **few lines** to start the app.


## Links
* The github repository is [here](https://github.com/unpackAI/tai-chi)