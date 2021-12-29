from tai_chi_engine import TaiChiTrained
from pathlib import Path
import logging
import pandas as pd
from PIL import Image
import streamlit as st

@st.cache(allow_output_mutation=True)
def load_trained(project):
    PROJECT = Path(project)
    logging.warning('loading... takes time')
    trained = TaiChiTrained(PROJECT)
    return trained

trained = load_trained("./project/image_regression")

slug = trained.phase['task_slug']
st.title('A Tai-Chi Engine Model:')
st.write(f'Task: {slug}')

from tai_chi_engine.quantify import (
    QuantifyText,
    QuantifyCategory,
    QuantifyImage,
    QuantifyMultiCategory,
    QuantifyNum
)
from typing import List, Any

def st_num(name, quantify, enrich=None):
    value = quantify.mean_
    min_value = quantify.mean_ - 3*quantify.std_
    max_value = quantify.mean_ + 3*quantify.std_
    
    return st.slider(
        label = name,
        min_value = min_value,
        max_value = max_value,
        value = value)
    
def st_image(name, quantify,enrich=None):
    if enrich is None:
        size = 224
        convert = "RGB"
    elif enrich['enrich']!="EnrichImage":
        size = 224
        convert = "RGB"
    else:
        size = enrich['kwargs']['size']
        convert = enrich['kwargs']['convert']
    uploaded_file = st.file_uploader(label=name, type=["jpg","png","jpeg","JPG","PNG","JPEG"])
        
    if uploaded_file is not None:
        image = Image.open(uploaded_file).resize((size, size))
        st.image(image, channels = convert)
        return image

def st_text(name, quantify, enrich=None):
    if quantify.max_length>64:
        return st.text_area(label = name, )
    else:
        return st.text_input(label = name)
        
def st_multiselect(name, quantify, enrich=None):
    return st.multiselect(label=name, options=quantify.category.i2c)

def st_select(name, quantify, enrich=None):
    return st.selectbox(label=name, options=quantify.category.i2c)

def get_enrich(trained,key):
    if 'enrich' not in trained.phase.config:
        return None
    for enrich in trained.phase['enrich']:
        if enrich['dst'] == key:
            return enrich
    return None


def build_app(trained):
    input_data = dict()
    for name, quantify in trained.qdict.items():
        if quantify.is_x:
            if type(quantify) == QuantifyNum:
                input_data.update({name: st_num(name, quantify)})
                logging.info(f"Input Field:{name} as float number")
            if type(quantify) == QuantifyImage:
                enrich = get_enrich(trained, name)
                input_data.update({name: st_image(name, quantify, enrich)})
                logging.info(f"Input Field:{name} as image")
            if type(quantify) == QuantifyText:
                logging.info(f"Input Field:{name} as text")
                input_data.update({name: st_text(name, quantify)})
            if type(quantify) == QuantifyMultiCategory:
                logging.info(f"Input Field:{name} as multi-select")
                val = st_multiselect(name, quantify)
                if quantify.actual_separator is not None:
                    val = quantify.actual_separator.join(val)
                input_data.update({name: val})
            if type(quantify) == QuantifyCategory:
                logging.info(f"Input Field:{name} as multi-select")
                input_data.update({name: st_select(name, quantify)})
            else:
                logging.warning(f"Don't know how to build Input Field:[{name}]")
                
    if st.button("Predict"):
        result = trained.predict(input_data)
        st.title("Result")
        if type(result)==pd.DataFrame:
            st.table(result.head(10))

build_app(trained)