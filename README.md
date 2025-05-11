# Hateful Memes Detection

This repository contains a multimodal approach to detect hate speech in memes for the Hateful Memes Challenge created by Facebook AI. The implementation focuses on early and late fusion techniques to effectively combine text and image modalities for improving hate speech detection performance.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Models](#models)
  - [Single Modality Models](#single-modality-models)
  - [Late Fusion Models](#late-fusion-models)
  - [Early Fusion Models](#early-fusion-models)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Grid Search](#grid-search)
  - [Evaluation](#evaluation)
  - [TensorBoard Visualization](#tensorboard-visualization)
- [Results](#results)
- [License](#license)

## Project Overview

The Hateful Memes Challenge is a dataset and benchmark to drive and measure progress on multimodal reasoning and understanding. The task focuses on detecting hate speech in multimodal memes, which requires both text and image understanding as well as reasoning about the relationship between the two modalities.

This implementation provides multiple approaches to address this challenge:

- Single modality models (text-only, image-only)
- Late fusion models (combining predictions from separate text and image models)
- Early fusion models (combining features before classification)

All models include TensorBoard integration, early stopping, model checkpointing, and grid search for hyperparameter optimization.

## Dataset

The dataset consists of memes with text and image modalities and binary labels indicating whether the meme is hateful (1) or not (0). The data files are in JSONL format:

```
data/
├── img/                  # Directory with meme images
├── train.jsonl           # Training set (8,500 samples)
├── dev.jsonl             # Development set (500 samples)
└── test.jsonl            # Test set (1,000 samples)
```

Each entry in the JSONL files contains:

- `id`: Unique identifier
- `img`: Path to the image file
- `text`: Text extracted from the meme
- `label`: Binary label (0 = not hateful, 1 = hateful)

## Architecture

The project is structured as follows:

```
hateful-memes-detection/
├── data/                 # Dataset directory
├── src/                  # Source code
│   ├── configs/          # Configuration files
│   ├── data/             # Data utilities
│   ├── models/           # Model implementations
│   │   ├── text_models.py           # Text-only models
│   │   ├── image_models.py          # Image-only models
│   │   ├── late_fusion_models.py    # Late fusion models
│   │   └── early_fusion_models.py   # Early fusion models
│   └── utils/            # Utility functions
│       ├── dataset.py               # Dataset loading
│       ├── trainer.py               # Training utilities
│       └── grid_search.py           # Grid search implementation
├── train.py              # Training script
├── grid_search_train.py  # Grid search script
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

## Models

### Single Modality Models

#### Text-Only Models

- **TextOnlyModel**: Uses BERT or other transformer models to process text with CLS token for classification
- **TextWithPoolingModel**: Extends text models with different pooling strategies (mean, max, cls)

#### Image-Only Models

- **ImageOnlyModel**: Uses CNN models (ResNet, EfficientNet, ViT) for image classification
- **ResNetWithAttention**: Implements spatial attention mechanism on feature maps

### Late Fusion Models

- **LateFusionModel**: Combines logits from separate text and image models using various methods:

  - Concatenation
  - Sum
  - Max
  - Weighted average

- **LateFusionWithAttention**: Uses learnable attention mechanism to combine modalities

### Early Fusion Models

- **EarlyFusionModel**: Combines features from text and image models before classification using:

  - Concatenation
  - Sum
  - Bilinear fusion

- **CrossAttentionFusion**: Implements cross-attention between text and image features

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/hateful-memes-detection.git
cd hateful-memes-detection
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Place the dataset files in the `data/` directory.

## Usage

### Training

To train a model, use the `train.py` script with appropriate arguments:

```bash
python src/train.py \
  --model_type late_fusion \
  --text_model_name bert-base-uncased \
  --img_model_name resnet50 \
  --fusion_method concat \
  --text_pooling_type cls \
  --batch_size 32 \
  --num_epochs 50 \
  --learning_rate 2e-5 \
  --early_stopping_patience 5
```

Key arguments:

- `--model_type`: Type of model to train (`text_only`, `image_only`, `late_fusion`, `early_fusion`, `cross_attention`)
- `--text_model_name`: Name of the pretrained text model (e.g., `bert-base-uncased`, `distilbert-base-uncased`)
- `--img_model_name`: Name of the pretrained image model (e.g., `resnet50`, `efficientnet_b0`)
- `--fusion_method`: Method for fusion (`concat`, `sum`, `max`, `weighted`, `bilinear`)
- `--num_epochs`: Number of epochs to train
- `--early_stopping_patience`: Number of epochs to wait for improvement before stopping

See all available options:

```bash
python src/train.py --help
```

### Grid Search

To perform grid search for hyperparameter tuning:

```bash
python src/grid_search_train.py \
  --model_type late_fusion \
  --num_epochs 3 \
  --batch_size 32
```

You can also provide a custom grid search configuration:

```bash
python src/grid_search_train.py \
  --model_type late_fusion \
  --grid_config configs/late_fusion_grid.json
```

Example grid configuration file:

```json
{
  "text_model_name": ["bert-base-uncased", "distilbert-base-uncased"],
  "img_model_name": ["resnet50", "efficientnet_b0"],
  "fusion_method": ["concat", "sum", "weighted"],
  "learning_rate": [1e-5, 2e-5, 5e-5],
  "dropout": [0.1, 0.3, 0.5],
  "weight_decay": [0.0, 0.01]
}
```

### Evaluation

Models are automatically evaluated on the validation set during training. The best model checkpoint is saved and can be used for inference. The evaluation uses AUROC as the primary metric, as recommended for the challenge.

### TensorBoard Visualization

Training progress can be visualized using TensorBoard:

```bash
tensorboard --logdir outputs/tensorboard
```

This will show training and validation metrics including loss, accuracy, and AUROC over time.

## Results

Grid search results and the best hyperparameters for each model type are saved in the `outputs/grid_search` directory. Training results, including model checkpoints and TensorBoard logs, are saved in the `outputs` directory.

The best performing models are typically:

- Late fusion models with attention or weighted fusion
- Early fusion models with cross-attention
- For single modality, text-only models tend to outperform image-only models

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite the original Hateful Memes Challenge paper:

```
@inproceedings{Kiela2020TheHM,
  title={The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes},
  author={Douwe Kiela and Hamed Firooz and Aravind Mohan and Vedanuj Goswami and Amanpreet Singh and Pratik Ringshia and Davide Testuggine},
  year={2020}
}
```
