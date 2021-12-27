__all__ = ["Quantify", "QuantifyImage", "QuantifyText",
           "QuantifyCategory", "QuantifyMultiCategory", "QuantifyNum"]

from tai_chi_tuna.front.html import Flash
from tai_chi_tuna.front.typer import STR, INT, BOOL, LIST

from tai_chi_tuna.flow.to_quantify import (
    BATCH_SIZE, SEQUENCE_SIZE, IMAGE_SIZE)
from tai_chi_engine.utils import clean_name

from forgebox.category import Category
from typing import List
import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path


class Quantify:
    is_quantify = True
    """
    # From all things to number
    The AI model does not understand anything, say, picture, text
    Unless you transform it to integer and float tensors

    Quantify and its subclass controls the
        numericalization / collation of the data pipeline
    The base class of quantify does: NOTHING
    """

    def __init__(self,):
        pass

    def __call__(self, list_of_items):
        return list(list_of_items)

    @staticmethod
    def quantify_dir(project: Path) -> Path:
        """
        Quantify directory under project directory
        """
        project = Path(project)
        quantify = project/'quantify'
        if not quantify.exists():
            quantify.mkdir(exist_ok=True, parents=True)
        return quantify

    @staticmethod
    def directory(project: Path, src: str) -> Path:
        """
        The directory to save the quantify column
        """
        save_dir = Quantify.quantify_dir(project)/clean_name(src)
        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True, parents=True)
        return save_dir

    def adapt(self, column):
        """
        A function to let the data processing
        adapt to the data column
        """
        pass

    def __hash__(self,):
        if hasattr(self, "name"):
            return self.name
        else:
            return self.__class__.__name__

    def save(self, project: Path, src: str):
        pass

    @classmethod
    def load(cls, project: Path, src: str):
        return cls()


class QuantifyImage(Quantify):
    """
    Transform PIL.Image to tensor
    """

    def __init__(
        self,
        mean_: LIST(["imagenet", "0.5 x 3"]) = "imagenet",
        std_: LIST(["imagenet", "0.5 x 3"]) = "imagenet",
    ):
        if type(mean_) == str:
            if mean_ == "imagenet":
                mean_ = [0.485, 0.456, 0.406]
            elif mean_ == "0.5 x 3":
                mean_ = [.5, .5, .5]
            else:
                raise ValueError(
                    f"Mean configuration: {mean_} not valid")

        if type(std_) == str:
            if std_ == "imagenet":
                std_ = [0.229, 0.224, 0.225]
            elif std_ == "0.5 x 3":
                std_ = [.5, .5, .5]
            else:
                raise ValueError(
                    f"Standard Variation configuration: {std_} not valid")

        from torchvision import transforms as tfm
        self.transform = tfm.Compose([
            tfm.ToTensor(),
            tfm.Normalize(mean=mean_, std=std_),
        ])
        self.mean_ = mean_
        self.std_ = std_

        self.shape = (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)

    def __repr__(self):
        return f"Quantify Image to tensors:{self.transform}"

    def __call__(self, list_of_image):
        return torch.stack(list(
            self.transform(img) for img in list_of_image))

    def save(self, project: Path, src: str):
        save_dir = self.directory(project, src)
        with open(save_dir/"config.json", 'w') as f:
            f.write(json.dumps({
                "mean": self.mean_,
                "std": self.std_,
            }, indent=2))

    @classmethod
    def load(cls, project: Path, src: str):
        save_dir = cls.directory(project, src)
        with open(save_dir/"config.json", 'r') as f:
            config = json.loads(f.read())
        return cls(mean_=config["mean"], std_=config["std"])
        


class QuantifyText(Quantify):
    def __init__(
        self,
        pretrained: STR(default="bert-base-cased") = "bert-base-cased",
        max_length: INT(default=512, min_=12, max_=1024, step=4) = 512,
        padding: LIST(options=[
            "do_not_pad",
            "max_length",
            "longest"], default="max_length") = "max_length",
        return_token_type_ids: BOOL(
            name="Token Type IDs", default=True) = True,
        return_attention_mask: BOOL(
            name="Attention Mask", default=True) = True,
        return_offsets_mapping: BOOL(
            name="Offset Mapping", default=False) = False,
    ):
        self.pretrained = pretrained
        self.max_length = max_length
        self.padding = padding
        self.return_token_type_ids = return_token_type_ids
        self.return_attention_mask = return_attention_mask
        self.return_offsets_mapping = return_offsets_mapping
        self.truncation = True
        self.return_tensors = 'pt'
        self.shape = (BATCH_SIZE, SEQUENCE_SIZE)

    def adapt(self, column):
        """
        Initialize tokenizer
        """
        from transformers import AutoTokenizer
        Flash.info("Loading transformer tokenizer, takes time", key="Alert!")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained, use_fast=True)

    def __call__(self, list_of_text: List[str]):
        list_of_text = list(list_of_text)
        return self.tokenizer(
            list_of_text,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
            return_token_type_ids=self.return_token_type_ids,
            return_attention_mask=self.return_attention_mask,
            return_tensors=self.return_tensors,
            return_offsets_mapping=self.return_offsets_mapping,
        )

    def save(self, project: Path, src: str) -> None:
        save_dir = self.directory(project, src)
        with open(save_dir/"config.json", 'w') as f:
            f.write(json.dumps({
                "pretrained": self.pretrained,
                "max_length": self.max_length,
                "padding": self.padding,
                "return_token_type_ids": self.return_token_type_ids,
                "return_attention_mask": self.return_attention_mask,
                "return_offsets_mapping": self.return_offsets_mapping,
            }, indent=2))

    @classmethod
    def load(cls, project: Path, src: str):
        save_dir = cls.directory(project, src)
        with open(save_dir/"config.json", 'r') as f:
            config = json.loads(f.read())
        obj = cls(
            pretrained=config["pretrained"],
            max_length=config["max_length"],
            padding=config["padding"],
            return_token_type_ids=config["return_token_type_ids"],
            return_attention_mask=config["return_attention_mask"],
            return_offsets_mapping=config["return_offsets_mapping"],
        )
        # load the tokenizer
        obj.adapt(None)
        return obj


class QuantifyCategory(Quantify):
    """
    Transform single categorical data to index numbers in pytorch tensors
    """

    def __init__(
        self,
        min_frequency: INT(min_=1, max_=20, default=1) = 1,
    ):
        self.min_frequency = min_frequency

    def adapt(self, column):
        # category statistics
        value_counts = pd.DataFrame(column.value_counts())

        # if minimun freq is 1
        # very category occured should be accounted for
        # hence no missing token padding is required
        if self.min_frequency < 2:
            self.category = Category(
                arr=np.array(value_counts.index),
                pad_mst=False)

        # we need missing token
        # for category's frequency < self.min_frequency
        else:
            categories = np.array(
                list(value_counts.index[
                    value_counts.values.reshape(-1) > self.min_frequency]))
            self.category = Category(arr=categories, pad_mst=True)

        self.shape = (BATCH_SIZE, len(self.category))

    def __repr__(self):
        return f"Quantify Category:{self.category}"

    def __call__(self, list_of_strings):
        return torch.LongTensor(self.category.c2i[np.array(list_of_strings)])

    def save(self, project: Path, src: str):
        save_dir = self.directory(project, src)
        self.category.save(save_dir/"category.json")
        with open(save_dir/"config.json", 'w') as f:
            f.write(json.dumps({
                "min_frequency": self.min_frequency,
            }, indent=2))
        
    @classmethod
    def load(cls, project: Path, src: str):
        save_dir = cls.directory(project, src)
        with open(save_dir/"config.json", 'r') as f:
            config = json.loads(f.read())
        obj = cls(min_frequency=config["min_frequency"])
        obj.category = Category.load(save_dir/"category.json")
        obj.shape = (BATCH_SIZE, len(obj.category))
        return obj


class QuantifyMultiCategory(Quantify):
    """
    Transform multi-categorical data to index numbers in pytorch tensors
    """

    def __init__(
        self,
        min_frequency: INT(min_=1, max_=20, default=1) = 1,
        separator: LIST(options=["[None]", ",", ";",
                        "[Space]", "[By Char]"], default=",") = ",",
    ):
        self.min_frequency = min_frequency
        friendly_mapping = {
            "[None]": None,
            "[Space]": " ",
            "[By Char]": "",
        }
        if separator in friendly_mapping:
            separator = friendly_mapping.get(separator)
        self.separator = separator

    @staticmethod
    def stripping(x):
        return x.strip()

    def break_cell(self, value):
        if value is None:
            return []
        break_list = list(i for i in map(
            self.stripping, str(value).split(self.separator)) if len(i) > 0)
        return break_list

    def adapt(self, column):
        if self.separator is None:
            sample_col = column
        else:
            sample_col = column.apply(self.break_cell)
        value_counts = pd.DataFrame(sample_col.explode().value_counts())

        # if minimun freq is 1
        # very category occured should be accounted for
        # hence no missing token padding is required
        if self.min_frequency < 2:
            self.category = Category(
                arr=np.array(value_counts.index),
                pad_mst=False)

        # we need missing token
        # for category's frequency < self.min_frequency
        else:
            categories = np.array(
                list(value_counts.index[
                    value_counts.values.reshape(-1) > self.min_frequency]))
            self.category = Category(arr=categories, pad_mst=True)

    def __call__(
        self, list_of_strings: List[str]
    ) -> torch.LongTensor:
        """
        Return a batch of n-hot array tensor
        """
        if self.separator is None:
            col: List[str] = list_of_strings
        else:
            col: List[List[str]] = list(map(self.break_cell, list_of_strings))
        arrays: List[np.array] = []
        for item in col:
            array: np.array = np.zeros(len(self.category))
            if len(item) > 0:
                one_idx: np.array = self.category.c2i[item]
                array[one_idx] = 1
            arrays.append(array)
        return torch.LongTensor(np.stack(arrays))

    def save(self, project: Path, src: str):
        save_dir = self.directory(project, src)
        self.category.save(save_dir/"category.json")
        with open(save_dir/"config.json", 'w') as f:
            f.write(json.dumps({
                "min_frequency": self.min_frequency,
                "separator": self.separator,
            }, indent=2))

    @classmethod
    def load(cls, project: Path, src: str):
        save_dir = cls.directory(project, src)
        with open(save_dir/"config.json", 'r') as f:
            config = json.loads(f.read())
        obj = cls(
            min_frequency=config["min_frequency"],
            separator=config["separator"],
        )
        obj.category = Category.load(save_dir/"category.json")
        return obj


class QuantifyNum(Quantify):
    """
    Quantify contineous data, like float numbers
    The only process is normalization on the entire population
    """
    shape = (BATCH_SIZE, 1)

    def adapt(self, column):
        self.mean_ = column.mean()
        self.std_ = column.std()

    def __call__(self, list_of_num):
        return (torch.FloatTensor(list_of_num)[:, None]-self.mean_)/self.std_

    def backward(self, x):
        return x*self.std_+self.mean_

    def save(self, project: Path, src: str):
        save_dir = self.directory(project, src)
        with open(save_dir/"config.json", 'w') as f:
            f.write(json.dumps({
                "mean": self.mean_,
                "std": self.std_,
            }, indent=2))
    
    @classmethod
    def load(cls, project: Path, src: str):
        save_dir = cls.directory(project, src)
        with open(save_dir/"config.json", 'r') as f:
            config = json.loads(f.read())
        obj = cls()
        obj.mean_ = config["mean"]
        obj.std_ = config["std"]
        return obj
