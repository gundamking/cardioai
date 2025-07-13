# Data Directory

This directory contains all datasets and data files used in the Medical AI Training project.

## Directory Structure

```
data/
├── clinical-guidelines/          # Medical guidelines and documents
│   └── KDIGO-2022-Clinical-Practice-Guideline-for-Diabetes-Management-in-CKD.pdf
├── statlog+heart/               # Heart disease prediction dataset
│   ├── heart.dat                # Main dataset file
│   ├── heart.doc                # Dataset documentation
│   └── Index                    # Dataset index
├── echocardiogram/              # Echocardiogram survival dataset
│   ├── echocardiogram.data      # Main dataset file
│   ├── echocardiogram.names     # Feature descriptions
│   └── Index                    # Dataset index
├── additional-studies/          # Additional research datasets
│   ├── ctg-studies.csv         # Cardiotocography studies
│   └── NCT04925245.csv         # Clinical trial data
├── archives/                    # Compressed dataset files
│   ├── statlog+heart.zip       # Heart disease dataset archive
│   └── echocardiogram.zip      # Echocardiogram dataset archive
└── README.md                    # This file
```

## Dataset Descriptions

### 1. Heart Disease Prediction (Statlog)
- **Source**: UCI Machine Learning Repository
- **Samples**: 270 patients
- **Features**: 13 clinical attributes
- **Target**: Binary classification (heart disease presence/absence)
- **Usage**: `src/heart_disease_predictor.py`

### 2. Echocardiogram Survival
- **Source**: UCI Machine Learning Repository
- **Samples**: 132 patients post-heart attack
- **Features**: 13 echocardiogram measurements
- **Target**: 1-year survival prediction
- **Usage**: `src/echocardiogram_analyzer.py`

### 3. KDIGO Clinical Guidelines
- **Source**: KDIGO 2022 Clinical Practice Guidelines
- **Content**: Diabetes management in chronic kidney disease
- **Usage**: `src/medical_llm_trainer.py`

### 4. Additional Studies
- **CTG Studies**: Cardiotocography research data
- **NCT04925245**: Clinical trial dataset
- **Usage**: Available for additional research

## Data Access

All datasets are publicly available and properly cited in the main README.md file.

## Notes

- Large files (>100MB) are excluded from Git via `.gitignore`
- Dataset archives are kept for backup purposes
- Clinical guidelines are used for LLM fine-tuning
- All datasets include proper documentation and feature descriptions 