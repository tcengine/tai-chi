# Build Apps for Tai-Chi Engine Trained Model

> As long as you have the project directory, you can build app upon the trained model, easily.

> "勁斷意不斷 意斷神可接"

## For now we have the streamlit app building
> See the [Streamlit App](./streamlit) building source code

### Start the app 🚀
```python
from tai_chi_engine.app import StartStreamLit

# You'll have to pick a trained project folder, and assign a port
tc_app = StartStreamLit("./netflix", port = 8501)
tc_app.start()
```

There is no second step, the above code will apply to **ALL** the project built by Tai-Chi Engine.

### Stop the app 🔪
```python
tc_app.stop()
```

```shell
sudo pkill -f "streamlit"
```
