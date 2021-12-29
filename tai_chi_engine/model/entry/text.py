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

    def reduce_hidden_state(self, vec, attention_mask):
        """
        Mean pooling over the hidden state according to attention_mask
        """
        vec = vec*(attention_mask[...,None])
        return vec.sum(dim=1)/(attention_mask.sum(-1)[...,None])

    def forward(self, kwargs):
        outputs = self.model(**kwargs)
        if self.encoder_mode:
            # output vector
            return self.reduce_hidden_state(
                outputs.last_hidden_state, kwargs["attention_mask"])
        return outputs

    @classmethod
    def from_quantify(
        cls,
        quantify,
        name: STR(default="bert-base-uncased") = 'bert-base-uncased',
        encoder_mode: BOOL(default=True) = True,
    ):
        # load transformer for inference
        if quantify.is_inference:
            from transformers import AutoConfig, AutoModel
            config = AutoConfig.from_pretrained(name)
            model = AutoModel.from_config(config)
        # load transformer for training
        else:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(name)
        # substatiate the transformer model
        obj = cls(model)
        obj.name = name
        obj.encoder_mode = encoder_mode
        if encoder_mode:
            obj.out_features= model.config.hidden_size
        return obj