"""Example of how to inspect a dataset using the configs."""
from psycop_model_training.load import load_train_from_cfg, load_train_raw
from psycop_model_training.config.schemas import load_cfg_as_pydantic


def main():
    """Main."""
    config_file_name = "default_config.yaml"

    cfg = load_cfg_as_pydantic(config_file_name=config_file_name)
    df = load_train_raw(cfg=cfg)  # noqa pylint: disable=unused-variable

    df_filtered = load_train_from_cfg(cfg=cfg)  # noqa pylint: disable=unused-variable


if __name__ == "__main__":
    main()
