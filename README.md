# Delta

Preference-learning experiments built with PyTorch Lightning. The project trains a personalized delta model that adjusts a base reward model using user embeddings and preference data.

## Quick start
- Python 3.10+ recommended. Install in editable mode: `pip install -e .`
- Install a matching PyTorch build for your CUDA/CPU setup (see https://pytorch.org/get-started/locally/).
- Launch training (example uses the bundled SimplER data):
	```bash
	python src/train.py \
		--dts_name simpler \
		--dts_config_file configs/data_configs.yaml \
		--config_file configs/exp_config.yaml \
		--batch_size 32 \
		--exp_name default_exp \
		--exp_version v0
	```
- Metrics and TensorBoard logs are written to `lightning_logs/<exp_name>/<version>/`.

## Data
- Datasets are described in [configs/data_configs.yaml](configs/data_configs.yaml). Supported keys: `simpler`, `prism`, `psoup`.
- Each dataset section declares a `dts_path` plus per-split files:
	- `df_file`: JSONL/Parquet with columns `prompt`, `chosen`, `rejected` (plus optional metadata such as `topic`, `qid`, and user features prefixed as listed in `prefix_columns`).
	- `text_file`: JSON array containing the unique texts used to align with precomputed embeddings.
	- `emb_file`: NumPy array of embeddings aligned to `text_file`.
- Provide the dataset name with `--dts_name` and the config file path with `--dts_config_file`.

## Experiment configuration
- Global trainer and model hyperparameters live in [configs/exp_config.yaml](configs/exp_config.yaml). Key sections:
	- `trainer`: epochs, batch size, optimizer, and scheduler knobs (see [src/delta/configs/trainer.py](src/delta/configs/trainer.py)).
  - `caimira`: default preference model based on item/user embeddings; `n_dim` can be overridden via `--n_dim`.
  - `model1`: alternative model enabled with `--model_name model1`; set `--features all` to include user feature vectors.
- Additional CLI switches in [src/train.py](src/train.py):
  - `--model_name`: select model to train (`caimira` or `model1`; defaults to `caimira`).

## Project layout
- [src/train.py](src/train.py): training entrypoint wiring configs, model, and data modules.
- [src/delta/lit_module.py](src/delta/lit_module.py): Lightning module and data module for preference training/testing.
- [configs/](configs): experiment and dataset configs.
- [scripts/data_preparation/](scripts/data_preparation): dataset splitting and embedding generation helpers.
- [data/simpler/](data/simpler): example SimplER dataset artifacts (JSONL, embeddings, texts).

## Tips
- Use TensorBoard to monitor training: `tensorboard --logdir lightning_logs`.
- Set `--features all` to include user features (only meaningful when using the `model1` model).
- Checkpoints can be configured via PyTorch Lightning callbacks if you want to persist best-performing models.
