# Tai-Chi Engine

[![PyPI version](https://img.shields.io/pypi/v/tai-chi-engine)](https://pypi.org/project/tai-chi-engine/)
![Python version](https://img.shields.io/pypi/pyversions/tai-chi-engine)
![License](https://img.shields.io/github/license/tcengine/tai-chi)
![PyPI Downloads](https://img.shields.io/pypi/dm/tai-chi-engine)
[![Docs](https://readthedocs.org/projects/tai-chi-engine/badge/?version=latest)](https://tai-chi-engine.readthedocs.io/en/latest/)
[![Test](https://github.com/tcengine/tai-chi/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/tcengine/tai-chi/actions/workflows/python-package-conda.yml)
[![pypi build](https://github.com/tcengine/tai-chi/actions/workflows/publish.yml/badge.svg)](https://github.com/tcengine/tai-chi/actions/workflows/publish.yml)

> Powerful deep learning for civilians

> ```太极引擎```  深度学习： 强大、多模态、灵活、平民化

> See [Documentation](https://tai-chi-engine.readthedocs.io/en/latest/) 看文档

## Essence of the Tai-Chi Engine
* Close to state-of-the-art Deep learning, friendly to office folks, all coding-free, clicks away.
* Flexible, supporting multiple kinds of data (image, text, category, multi-category, etc), multiple x at the same time.
* Which columns to be x? Which column to be y? You can decide and play, see if the AI finds out how to guess.

## Our big pitch 🎁
* If you're a **coding muggle** - play with the engine, you'll understand ideas around Deep Learning, and have good model.
* If you're a **pro** - it's still wildly fun to try new models in 2 minutes' click, especially you have around a dozen columns.

## Playing Tai-Chi Engine
First, tell me you are already in a jupyter notebook environment.

Or just using the free kaggle kernel, [example here](https://www.kaggle.com/raynardj/tc-engine-versatility)

Or try our colab tutorial: 
<p><a href="https://colab.research.google.com/github/tcengine/tai-chi/blob/main/nbs/tai_chi_colab.ipynb"><img data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667"></a></p>

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

Good to go! 🎸

## Installation 📦
Default installation:

```bash
pip install tai-chi-engine
```

## Run App on Trained Model 🚀
You can build prototype App (Based on [streamlit](https://streamlit.io/)) based on trained project, see [app](https://github.com/tcengine/tai-chi/tree/main/tai_chi_engine/app) part of this library.

ALL kinds of projects, only **one way** and only **few lines** to start the app.


## Links 🪁
* The github repository is [here](https://github.com/tcengine/tai-chi)

## For Developers:
* How frontend part of the engine works? [See here](nbs/frontend.ipynb)