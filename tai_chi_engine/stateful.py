__all__ = ["Stateful", "STATE_TYPE"]

from pathlib import Path
from tai_chi_engine.utils import clean_name
import json


class StateType:
    def __init__(self, type_):
        self.type_ = type_

    def __repr__(self):
        return f"{self.type_}"

    def load(self, save_dir, value):
        return self.type_(value)

    def save(self, save_dir, value):
        """
        Return the value that can be saved as JSON string
        """
        return value


class LoadCategory(StateType):
    def __init__(self):
        super().__init__("Category")

    def __repr__(self):
        return f"{self.filename}"

    def load(self, save_dir, value):
        from category import Category
        return Category.load(save_dir/value)

    def save(self, save_dir, value) -> str:
        value.save(save_dir/"category.json")
        return "category.json"


class LoadPretrainedTokenizer(StateType):
    def __init__(self):
        super().__init__("Transformers.Tokenizer")

    def __repr__(self):
        return f"{self.filename}"

    def load(self, save_dir, value):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(save_dir/value), use_fast=True)
        return tokenizer

    def save(self, save_dir, value) -> str:
        pretrained = "tokenizer"
        from transformers import AutoTokenizer
        value.save_pretrained(str(save_dir/pretrained))
        return pretrained


STATE_TYPE = dict({
    "str": StateType(str),
    "int": StateType(int),
    "float": StateType(float),
    "bool": StateType(bool),
    "list": StateType(list),
    "category": LoadCategory(),
    "tokenizer": LoadPretrainedTokenizer(),
})


class Stateful:
    phase_state = "tai-chi-engine"
    stateful_conf = dict()

    @classmethod
    def phase_state_dir(cls, project: Path) -> Path:
        """
        phase state directory under project directory
        """
        project = Path(project)
        phase_state = project/cls.phase_state
        if not phase_state.exists():
            phase_state.mkdir(exist_ok=True, parents=True)
        return phase_state

    @classmethod
    def save_dir(cls, project: Path, name: str) -> Path:
        """
        The directory to save the column
        """
        save_dir = cls.phase_state_dir(project)/clean_name(name)
        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True, parents=True)
        return save_dir

    def save(self, project: Path, name: str):
        """
        default save function
        """
        save_dir = self.save_dir(project, name)
        to_save_data = dict()
        # iter through property name and property type
        for property_name, state_type_key in self.stateful_conf.items():
            running_value = getattr(self, property_name)
            state_type = STATE_TYPE[state_type_key]
            save_value = state_type.save(save_dir, running_value)
            to_save_data[property_name] = save_value
        with open(save_dir/"config.json", "w") as f:
            json.dump(to_save_data, f)

    @classmethod
    def load_conf(cls, project: Path, name: str):
        """
        Load object from configuration file
        """
        save_dir = cls.save_dir(project, name)
        config_path = save_dir/'config.json'
        if config_path.exists():
            with open(save_dir/"config.json", 'r') as f:
                config = json.loads(f.read())
        else:
            config = dict()
        return config

    @classmethod
    def load(cls, project: Path, name: str):
        """
        Load from configuration file
        """
        save_dir = cls.save_dir(project, name)
        config = cls.load_conf(project, name)

        # property data for __init__ function
        init_dict = dict()
        # property data to set attribute after __init__
        setattr_dict = dict()
        # translate the config to the stateful object
        for property_name, state_type_key in cls.stateful_conf.items():
            state_type = STATE_TYPE[state_type_key]
            running_value = state_type.load(
                save_dir, config[property_name])

            if property_name in cls.__init__.__annotations__:
                init_dict[property_name] = running_value
            else:
                setattr_dict[property_name] = running_value
        obj = cls(**init_dict)
        for property_name, running_value in setattr_dict.items():
            setattr(obj, property_name, running_value)
        return obj
