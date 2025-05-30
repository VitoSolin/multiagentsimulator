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

1. **Download from original source**: Avazu Click-Through Rate Prediction dataset
2. **Contact repository owner** for preprocessed files
3. **Use sample data** in `multiagent/` directory for testing

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