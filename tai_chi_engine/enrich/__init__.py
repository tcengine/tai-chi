__all__ = ["EnrichImage", "ParentAsLabel", "ENRICHMENTS"]


from .basic import EnrichCleanTyping
from .file import (
    EnrichImage, ParentAsLabel
    )

ENRICHMENTS = dict(
    EnrichImage=EnrichImage,
    ParentAsLabel=ParentAsLabel,
    CleanData=EnrichCleanTyping
)