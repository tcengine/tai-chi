__all__ = ['TransformerEncoder']

from tai_chi_tuna.front.typer import BOOL, STR
from .basic import EntryModel

class TransformerEncoder(EntryModel):
    """
    A model part to encode sequnce data in to vectors
    """

    def __init__(self, model, encoder_mode: BOOL(default=True) = True,):
        super().__init__()
        self.model = model
        self.encoder_mode = encoder_mode

    def forward(self, kwargs):
        outputs = self.model(**kwargs)
        if self.encoder_mode:
            # output vector
            if "pooler_output" in outputs:
                return outputs.pooler_output
            else:
                return (
                    outputs.last_hidden_state*kwargs['attention_mask'][:,:,None]
                ).mean(1)
        return outputs

    @classmethod
    def from_quantify(
        cls,
        quantify,
        name: STR(default="bert-base-uncased") = 'bert-base-uncased',
        encoder_mode: BOOL(default=True) = True,
    ):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(name)
        obj = cls(model)
        obj.name = name
        obj.encoder_mode = encoder_mode
        if encoder_mode:
            obj.out_features= model.config.hidden_size
        return obj