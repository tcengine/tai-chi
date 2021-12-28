from tai_chi_tuna.config import PhaseConfig
from tai_chi_tuna.flow.to_quantify import load_qdict
from tai_chi_tuna.flow.to_model import assemble_model

from .enrich import ENRICHMENTS
from .quantify import QUANTIFY
from .model import (ALL_EXIT, QUANTIFY_2_EXIT_MAP,
                    ALL_ENTRY, QUANTIFY_2_ENTRY_MAP)
import torch
import logging
from pathlib import Path
from typing import Dict, Any


class TaiChiTrained:
    """
    Trained project
    """
    def __init__(self, project: Path, device:str = "cpu"):
        """
        Load a trained project from a project directory
        """
        self.project = Path(project)
        if self.project.exists() == False:
            raise FileNotFoundError(
                f"Project {self.project} does not exist")
        self.phase = PhaseConfig.load(project)
        self.load_things()
        self.device = torch.device(device)

    def __repr__(self):
        return f"[☯️ Project: {self.project}]\n" +\
            "\tmodel:\tself.final_model\n" +\
            "\tquantify:\tself.qdict\n" +\
            f"\tx_columns:\t{self.x_columns}\n" +\
            f"\ty_columns:\t{self.y_columns}\n"

    @property
    def best_checkpoint(self,):
        checkpoints = list((self.project/"checkpoints").glob("*.ckpt"))
        if len(checkpoints) == 0:
            raise FileNotFoundError(
                f"No checkpoints found in {self.project}")
        return checkpoints[-1]

    def to_tensor(self, data: Dict[str, Any]):
        """
        Convert the data to tensor
        """
        tensor_data = dict()
        for k in self.x_columns:
            value = data[k]
            tensor_data[k] = self.qdict[k](list([value,]))
            tensor_data[k].to(self.device)
        return tensor_data

    def predict(self, data):
        tensor_data = self.to_tensor(data)
        with torch.no_grad():
            pred = self.final_model.eval_forward(tensor_data)
        return self.y_quantify.backward(pred[0])

    def load_things(self):
        module_zoo = {"all_entry": ALL_ENTRY, "all_exit": ALL_EXIT}
        self.qdict = load_qdict(project=self.project,
                                phase=self.phase, quantify_map=QUANTIFY)

        self.final_model = assemble_model(
            phase=self.phase, qdict=self.qdict, modules=module_zoo)
            
        logging.info(f"Loaded model from {self.best_checkpoint}")
        state_dict = torch.load(str(self.best_checkpoint), map_location="cpu")
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        self.final_model.load_state_dict(state_dict)
        self.final_model.eval()

        self.x_columns = list(
            quantify['src']
            for quantify in self.phase['quantify'] if quantify['x'])

        self.y_columns = list(
            quantify['src']
            for quantify in self.phase['quantify'] if quantify['x']==False)

        self.y_quantify = self.qdict[self.y_columns[0]]
