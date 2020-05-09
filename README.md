# Templates collection for torch-points3

## Setup

Install the dependencies with poetry

```
pip install poetry
poetry install
poetry shell
```

## Pytorch Lighthing

This script runs a `point cloud classifier` with several bakcbones on ModelNet under `100 lines`.

The bakcbones are `"kpconv", "pointnet2", "rsconv"`

```
poetry run python3 examples/pytorch-lightning/main.py kpconv
```

## FastAI

This script runs a `point cloud Unet segmentation model` with several bakcbones on Shapenet under `100 lines`.

The bakcbones are `"kpconv", "pointnet2", "rsconv"`

```
poetry run python3 examples/fastai/main.py rsconv
```
