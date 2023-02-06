from pathlib import Path
from types import NoneType
from typing import List

from psycop_model_training.config_schemas.full_config import FullConfigSchema
from src.psycop_model_training.application_modules.wandb_handler import WandbHandler


def test_wandb_handler_fullconfig_parsing(
    muteable_test_config: FullConfigSchema,
):
    """Test that the wandb handler can parse a fullconfig object to a flattened
    dict, ready for upload to wandb."""
    cfg_parsed = WandbHandler(
        cfg=muteable_test_config
    )._get_cfg_as_dict()  # pylint: disable=protected-access

    for k, v in cfg_parsed.items():
        if not isinstance(k, str):
            raise AssertionError(f"Key {k} is not of the correct type.")
        if not isinstance(v, (str, int, float, NoneType, Path, list)):
            raise AssertionError(f"Value {v} in config is not of the correct type.")
