import wandb

run = wandb.init(
    project="Project-Build-an-ML-Pipeline-Starter",
    entity="gberha1-western-governors-university",
    job_type="download_artifact"
)

artifact = run.use_artifact("gberha1-western-governors-university/Project-Build-an-ML-Pipeline-Starter-components_train_val_test_split/trainval_data.csv:v0", type="trainval_data")
  
artifact_dir = artifact.download()

print(f"Artifact downloaded to: {artifact_dir}")
