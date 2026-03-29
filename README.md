# Plant seed competition 

## Overview 
The objective is to classify a plant seeding into one of 12 species. 
**Competition:** [Plant Seedling Classification](https://www.kaggle.com/c/plant-seedling-classification)

## Approach 
- **Model:** EfficientNet-B3 pretrained on ImageNet (transfer learning)
- **Validation:** 5-Fold Stratified Cross Validation
- **Augmentations:** Horizontal/Vertical Flip, RandomRotate90, ShiftScaleRotate, ColorJitter
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler:** Cosine Annealing with Warm Restarts (T_0=10, warmup=2 epochs)
- **Loss:** CrossEntropy with Label Smoothing (0.1)


## Project Structure 
plant_seeding_classification/
│
├── data/                     
│   ├── train/
│   └── test/
│
├── notebooks/                
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_experiments.ipynb
│
├── src/                      
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── transforms.py
│   ├── train.py
│   ├── inference.py
│   ├── submission.py
│   └── utils.py
│
├── outputs/                  
│   ├── models/
│   ├── oof/
│   ├── submissions/
│   └── logs/
│
├── knowledge/                
│
├── configs/                  
│   └── exp001.yaml
│
├── scripts/                  
│   ├── train.sh
│   └── inference.sh
│
├── .gitignore                
├── requirements.txt          
├── README.md                



## Setup
```bash
   pip install -r requirements.txt
```


## Key techniques 
- Transfer Learning (Efficient-B3)
- Stratified KFold Cross validation
- Cosine Annealing with Warm Restarts 
- Albumentations augmentation pipeline

