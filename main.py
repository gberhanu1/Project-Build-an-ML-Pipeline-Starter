import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig,OmegaConf

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_path="config", config_name="config", version_base=None)
def go(config: DictConfig):
    OmegaConf.set_struct(config, False)

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                uri=os.path.abspath("./components/train_val_test_split"),
                entry_point="main",
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw_file_as_downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": config["etl"]["cleaning_input_artifact"],
                    "output_artifact": config["etl"]["cleaning_output_artifact"],
                    "output_type": config["etl"]["cleaning_output_type"],
                    "output_description": config["etl"]["cleaning_output_description"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                },
            )
        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                "main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["data_check"]["min_price"],
                    "max_price": config["data_check"]["max_price"]
                },
            )
            

        if "data_split" in active_steps:
            _ = mlflow.run(
                uri=os.path.abspath(os.path.join(hydra.utils.get_original_cwd(), "components", "train_val_test_split")),
                entry_point="main",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": config.modeling.test_size,
                    "random_seed": config.modeling.random_seed,
                    "stratify_by": config.modeling.stratify_by
                }
            )
        if "train_random_forest" in active_steps:
            from hydra.utils import to_absolute_path
            component_path = to_absolute_path("src/train_random_forest")
            # NOTE: we need to serialize the random forest configuration into JSON
            modeling_config = OmegaConf.to_container(config.modeling, resolve=True)
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            mlflow.run(
                component_path,
                entry_point="main",
                parameters={
                    "trainval_artifact": os.path.abspath("trainval_data.csv"),
                    "output_artifact": "random_forest_export",
                    "rf_config": rf_config,
                    "val_size": float(config.modeling.val_size),
                    "max_tfidf_features": int(config.modeling.max_tfidf_features),
                    "random_seed": int(config.modeling.random_seed),
                    "stratify_by": str(config.modeling.stratify_by)
                }
                
            )
            

        if "test_regression_model" in active_steps:

            ##################
            # Implement here #
            ##################

            pass


if __name__ == "__main__":
    go()
