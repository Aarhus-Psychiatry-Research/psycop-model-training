"""Small script to remove the timezone from outcome timestamps that somehow appeared."""
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from psycop_model_training.config_schemas.conf_utils import (
    convert_omegaconf_to_pydantic_object,
)
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.data_loader.data_loader import DataLoader
from psycop_model_training.training.train_and_predict import CONFIG_PATH


@hydra.main(
    config_path=str(CONFIG_PATH),
    config_name="default_config",
    version_base="1.2",
)
def main(cfg: DictConfig):
    if not isinstance(cfg, FullConfigSchema):
        cfg = convert_omegaconf_to_pydantic_object(cfg)
    dataloader = DataLoader(cfg=cfg)

    for split_name in ["train", "val", "test"]:
        df = dataloader.load_dataset_from_dir(split_names=split_name)
        df = normalize_timestamp_column(
            df, timestamp_column=cfg.data.col_name.outcome_timestamp
        )
        save_split(cfg=cfg, df=df, split_name=split_name)


def normalize_timestamp_column(df: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """Remove the timezone from the timestamp column."""
    df[timestamp_column] = df[timestamp_column].dt.tz_localize(None)
    return df


def save_split(cfg: FullConfigSchema, split_name: str, df: pd.DataFrame):
    """Save the split."""
    data_dir = Path(cfg.data.dir)
    path = list(data_dir.glob(f"*{split_name}*.{cfg.data.suffix}"))[0].name
    save_path = data_dir / "timezone_normlized"
    save_path.mkdir(exist_ok=True)
    save_path = save_path / path
    print(f"Saving {save_path}")
    df.to_parquet(save_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
