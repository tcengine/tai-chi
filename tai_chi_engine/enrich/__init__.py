__all__ = ["EnrichImage", "ParentAsLabel", "ENRICHMENTS"]

from .file import (
    EnrichImage, ParentAsLabel
    )

ENRICHMENTS = dict(
    EnrichImage=EnrichImage,
    ParentAsLabel=ParentAsLabel,
)