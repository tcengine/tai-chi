from tai_chi_engine import TaiChiTrained
from tai_chi_engine.quantify import (
    QuantifyText,
    QuantifyCategory,
    QuantifyImage,
    QuantifyMultiCategory,
    QuantifyNum
)
import streamlit as st
from typing import Any, Dict, Callable
from pathlib import Path
import logging
import pandas as pd
from PIL import Image
import os


@st.cache(allow_output_mutation=True)
def load_trained(project: Path) -> TaiChiTrained:
    """
    Cache function to load medel
    """
    PROJECT = Path(project)
    logging.warning('loading... takes time')
    trained = TaiChiTrained(PROJECT)
    return trained


def get_enrich_map(trained: TaiChiTrained) -> Dict[str, Dict[str, Any]]:
    """
    Search the enrich configs by column names
    """
    if 'enrich' not in trained.phase.config:
        return dict()
    enrich_map = dict()
    for enrich in trained.phase['enrich']:
        enrich_map[enrich['dst']] = enrich
    return enrich_map


def to_build_st(QuantifyClass: type) -> Callable:
    """
    Assgin the QuantifyClass with build_st_field
    Use as decorator
    """
    def to_build_st_decorator(func: Callable) -> Callable:
        QuantifyClass.build_st_field = func
        return func
    return to_build_st_decorator


@to_build_st(QuantifyNum)
def st_num(quantify: QuantifyNum, name: str, enrich=None):
    """
    Build float number input slider
    """
    value = quantify.mean_
    min_value = quantify.mean_ - 3*quantify.std_
    max_value = quantify.mean_ + 3*quantify.std_

    return st.slider(
        label=name,
        min_value=min_value,
        max_value=max_value,
        value=value)


@to_build_st(QuantifyImage)
def st_image(quantify: QuantifyImage, name: str, enrich=None):
    """
    Build image upload field
    The result will be resized and converted
        according to the enrich config
    """
    if enrich is None:
        size = 224
        convert = "RGB"
    elif enrich['enrich'] != "EnrichImage":
        size = 224
        convert = "RGB"
    else:
        size = enrich['kwargs']['size']
        convert = enrich['kwargs']['convert']
    uploaded_file = st.file_uploader(
        label=name, type=["jpg", "png", "jpeg", "JPG", "PNG", "JPEG"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).resize((size, size))
        st.image(image, channels=convert)
        return image


@to_build_st(QuantifyText)
def st_text(quantify: QuantifyText, name: str, enrich=None):
    """
    Build text field based on max length
    if max length >64, use the text area
    if not, use the text input field
    """
    if quantify.max_length > 64:
        return st.text_area(label=name, )
    else:
        return st.text_input(label=name)


@to_build_st(QuantifyMultiCategory)
def st_multiselect(quantify: QuantifyMultiCategory, name: str, enrich=None):
    """
    Build multi-select based on the categories
    """
    val = st.multiselect(label=name, options=quantify.category.i2c)
    if quantify.actual_separator is not None:
        val = quantify.actual_separator.join(val)
    return val


@to_build_st(QuantifyCategory)
def st_select(quantify: QuantifyCategory, name: str, enrich=None):
    """
    Build select field for single categorical input
    """
    return st.selectbox(label=name, options=quantify.category.i2c)


def build_app(project: Path):
    """
    Build streamlit app
    with TaiChiTrained
    """
    # load a trained model
    trained = load_trained(Path(project))
    slug = trained.phase['task_slug']
    # build title
    st.markdown(f"""
    ## A Tai-Chi Engine Model:
    > Trained with [Tai-Chi Engine](https://github.com/tcengine/tai-chi)
    """)
    st.write(f'Task: {slug}')

    input_data = dict()
    enrich_map = get_enrich_map(trained)

    # build the input data
    for name, quantify in trained.qdict.items():
        # only build input upon x
        if quantify.is_x:
            if hasattr(quantify, 'build_st_field'):
                enrich = enrich_map.get(name)
                input_data[name] = quantify.build_st_field(
                    name, enrich)
            else:
                raise ValueError(
                    f"We don't have ways to build input field for column: {name}")

    # manage the predict output
    if st.button("Predict"):
        result = trained.predict(input_data)
        st.title("Result")
        if type(result) == pd.DataFrame:
            st.table(result.head(10))
        else:
            st.write(result)


if __name__ == "__main__":
    try:
        TAI_CHI_ENGINE_APP = os.environ['TAI_CHI_ENGINE_APP']
    except:
        print("Please set TAI_CHI_ENGINE_APP environment variable to the project directory")
        exit(1)

    build_app(TAI_CHI_ENGINE_APP)
