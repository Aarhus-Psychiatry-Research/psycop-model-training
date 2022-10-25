"""Test that the model trains correctly."""

import pytest
from hydra import compose, initialize

from psycopt2d.models import MODELS
from psycopt2d.train_model import main
from psycopt2d.utils.configs import omegaconf_to_pydantic_objects

CONFIG_DIR_PATH = "../src/psycopt2d/config/"
INTEGRATION_TEST_FILE_NAME = "integration_testing.yaml"
INTEGRATION_TESTING_MODEL_OVERRIDE = "model=logistic-regression"


@pytest.mark.parametrize("model_name", MODELS.keys())
def test_main(model_name):
    """test main using a variety of model."""
    with initialize(version_base=None, config_path=CONFIG_DIR_PATH):

        cfg = compose(
            config_name=INTEGRATION_TEST_FILE_NAME,
            overrides=[f"model={model_name}"],
        )

        cfg = omegaconf_to_pydantic_objects(cfg)

        # XGBoost should train on GPU on Overtaci,
        # but CPU during integration testing
        if model_name == "xgboost":
            cfg.model.args["tree_method"] = "auto"

        main(cfg)


@pytest.mark.pre_push_test
def test_integration_test():
    """test main using the logistic model.

    Used for quickly testing functions before a push.
    """
    with initialize(version_base=None, config_path=CONFIG_DIR_PATH):

        cfg = compose(
            config_name=INTEGRATION_TEST_FILE_NAME,
            overrides=[INTEGRATION_TESTING_MODEL_OVERRIDE],
        )
        main(cfg)


def test_crossvalidation():
    """Test crossvalidation."""
    with initialize(version_base=None, config_path=CONFIG_DIR_PATH):
        cfg = compose(
            config_name=INTEGRATION_TEST_FILE_NAME,
            overrides=[INTEGRATION_TESTING_MODEL_OVERRIDE, "+data.n_splits=2"],
        )
        main(cfg)


def test_min_prediction_time_date():
    """Test crossvalidation."""
    with initialize(version_base=None, config_path=CONFIG_DIR_PATH):
        cfg = compose(
            config_name=INTEGRATION_TEST_FILE_NAME,
            overrides=[
                INTEGRATION_TESTING_MODEL_OVERRIDE,
                "+data.min_prediction_time_date=1972-01-01",
            ],
        )
        main(cfg)


def test_feature_selection():
    """Test feature selection."""
    with initialize(version_base=None, config_path=CONFIG_DIR_PATH):

        cfg = compose(
            config_name=INTEGRATION_TEST_FILE_NAME,
            overrides=[
                INTEGRATION_TESTING_MODEL_OVERRIDE,
                "preprocessing.feature_selection.name=f_classif",
                "preprocessing.feature_selection.params.percentile=10",
            ],
        )

        main(cfg)
