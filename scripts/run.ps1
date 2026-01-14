$Image = "mlagents-pipeline"

# Repo root = parent of this script's folder (scripts\)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir "..")

docker build -t $Image $Root

docker run --rm -it --init `
  -e USER=$env:USERNAME -e LOGNAME=$env:USERNAME `
  -v "${Root}\training_manager\experiments\configs_for_training:/app/training_manager/experiments/configs_for_training" `
  -v "${Root}\training_manager\experiments\results:/app/training_manager/experiments/results" `
  $Image `
  python -m training_pipeline.cli.run_experiment $args
