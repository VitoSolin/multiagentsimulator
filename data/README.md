# Data Directory

## Large Dataset Files (Not in Git)

Due to file size limitations, the following large datasets are excluded from Git:

### Training Data
- `train.csv` (5.9GB) - Main training dataset
- `test.csv` (673MB) - Test dataset  
- `avazu_dev_pro.parquet` (21MB) - Processed development data

### Model Files
- `ctr_model_pro_two.txt` (2.3MB) - Trained CTR model
- Large model dumps and artifacts

## How to Get the Data

### Training Data (train.csv, test.csv)
Download from Kaggle Avazu CTR Prediction competition:
- **Link**: https://www.kaggle.com/c/avazu-ctr-prediction/data?select=train.gz
- Extract the downloaded files to this directory

### Preprocessed Files (avazu_dev_pro.parquet, ctr_model_pro_two.txt)
Download from Google Drive:
- **Link**: https://drive.google.com/drive/folders/1ZLSY7XI8Y-wc1i_eRV9z_CjqBGk82E5C?usp=sharing
- Download and place in this directory

### Alternative Options
- **Use sample data** in `multiagent/` directory for testing

## Setup Instructions

```bash
# 1. Download data files to this directory
# 2. Run preprocessing (if needed)
python ../multiagent/sample_and_convert.py

# 3. Train models
python ../not-neccessary/ctr_train.py
```

## Data Structure Expected

```
data/
├── train.csv              # Main training data
├── test.csv               # Test data  
├── avazu_dev_pro.parquet  # Processed data
├── ctr_model_pro_two.txt  # Trained model
└── README.md              # This file
``` 