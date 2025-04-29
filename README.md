# Transformer DE→EN Translation

This project implements a sequence‐to‐sequence Transformer for German→English translation, refactored into a `src/` package:

- `src/data/data_loader.py` – dataset download, tokenization, vocabulary, DataLoaders  
- `src/data/utils.py`       – mask creation & constants  
- `src/models/model.py`       – model definitions (embeddings, Transformer)  
- `src/models/train.py`       – training & evaluation loops  
- `src/data/inference.py`   – greedy decoding and translation  
- `src/main.py`        – CLI entrypoint for `train` and `translate`

## Installation

```bash
git clone <repo-url>
cd <repo-dir>
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

### Train
```bash
python -m src.main train --epochs 10 --batch-size 128
```

### Translate
```bash
python -m src.main translate "Ein kleines Mädchen klettert auf einen Baum."
```