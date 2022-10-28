"""Dataclasses for evaluation."""
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from psycopt2d.utils.configs import BaseModel, FullConfigSchema


class EvalDataset(BaseModel):
    """Evaluation dataset.

    Makes the interfaces of our evaluation functions simpler and
    consistent.
    """

    class Config:
        """Configuration of Pydantic model."""

        allow_mutation = True

    ids: pd.Series
    pred_timestamps: pd.Series
    outcome_timestamps: pd.Series
    y: pd.Series
    y_hat_probs: pd.Series
    y_hat_int: pd.Series
    age: Optional[pd.Series]


class ArtifactContainer(BaseModel):
    """A container for artifacts."""

    label: str
    # We're not a big fan of the naming here, super open to suggestions!
    # We need to keep the artifact and its labeled coupled, hence the
    # need for a container.
    artifact: Union[Path, pd.DataFrame]


class PipeMetadata(BaseModel):
    """Metadata for a pipe.

    Currently only has feature_importances, but makes it easy to add more - e.g. which features were dropped.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Config.allow_mutation = True

    feature_importances: Optional[dict[str, float]] = None


class ModelEvalData(BaseModel):
    """Dataclass for model evaluation data."""

    eval_dataset: EvalDataset
    cfg: FullConfigSchema
    pipe_metadata: Optional[PipeMetadata] = None
