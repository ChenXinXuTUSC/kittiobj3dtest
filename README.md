# SqueezeSeg

SqueezeSeg is a semantic segmentation framework with a focus on LiDAR point clouds and applications in autonomous driving, particularly with the KITTI dataset.

## Project Structure

The project is organized as follows:

-   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
-   `Makefile`: Automates build processes.
-   `README.md`: This file, providing an overview of the project.
-   `train.py`: Training script for the models.
-   `train.sh`: Shell script to run the training.
-   `test.py`: Testing script for the models.
-   `test.sh`: Shell script to run the testing.
-   `conf/`: Configuration files for datasets, losses, metrics, models, and pipelines.
    -   `dataset/`: Dataset configurations (e.g., `dataset_kittisem.toml`, `dataset_kittiobj3d.toml`).
    -   `loss/`: Loss function configurations (e.g., `loss_squeezeseg.toml`, `loss_deeplabv3.toml`).
    -   `metric/`: Metric configurations (e.g., `metric_deeplabv3.toml`).
    -   `model/`: Model configurations (e.g., `model_squeezeseg.toml`, `model_deeplabv3.toml`).
    -   `pipe/`: Pipeline configurations (e.g., `pipe_kittisem+deeplabv3.toml`).
-   `core/`: Core implementation of the framework.
    -   `conf/`: Configuration management.
    -   `dataset/`: Dataset classes (e.g., `dataset_kittisem.py`, `dataset_kittiobj3d.py`).
    -   `loss/`: Loss function implementations (e.g., `loss_squeezeseg.py`, `loss_deeplabv3.py`).
    -   `metric/`: Metric implementations (e.g., `metric_squeezeseg.py`, `metric_deeplabv3.py`).
    -   `model/`: Model implementations (e.g., `model_squeezeseg.py`, `model_deeplabv3.py`).
    -   `pipe/`: Training pipeline management.
-   `data/`: Directory for storing datasets.
-   `doc/`: Documentation.
-   `hub/`: Model hub related files.
-   `log/`: Training logs.
-   `notebook/`: Jupyter notebooks for experimentation.
-   `test/`: Test scripts.
-   `utils/`: Utility functions.

## Functionality

The framework provides the following functionalities:

-   **Semantic Segmentation**: Implements semantic segmentation algorithms, particularly for LiDAR point clouds.
-   **KITTI Dataset Support**: Specifically designed for the KITTI dataset, with configurations and dataset classes for KITTIObj3d and KITTISemantic.
-   **Configuration System**: Uses TOML files for configuring datasets, losses, metrics, models, and training pipelines.
-   **Training and Testing**: Includes scripts for training and testing models.
-   **Modular Design**: The framework is designed with a modular structure, allowing for easy extension and customization.

## Usage

To use the framework, follow these steps:

1.  Configure the dataset, loss, metric, and model in the `conf/` directory.
2.  Run the training script `train.py` to train the model.
3.  Run the testing script `test.py` to evaluate the model.
