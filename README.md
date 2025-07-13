# 🏥 Medical AI Training & Prediction System

A comprehensive healthcare AI system for medical text analysis and clinical prediction, developed during the Rutgers Health Hack 2024. This project combines Large Language Model (LLM) fine-tuning with traditional machine learning for medical applications.

## 🎯 Project Overview

This repository contains three main components:

1. **Medical LLM Fine-tuning** - Fine-tune Llama-2 on clinical guidelines (KDIGO diabetes management)
2. **Heart Disease Prediction** - ML models for cardiovascular risk assessment
3. **Echocardiogram Analysis** - Survival prediction after heart attacks

## 📊 Datasets

### 1. KDIGO Clinical Guidelines
- **Source**: KDIGO 2022 Clinical Practice Guideline for Diabetes Management in CKD
- **Purpose**: Fine-tune LLM for medical question answering
- **Format**: PDF document with clinical recommendations

### 2. Statlog Heart Disease Dataset
- **Samples**: 270 patients
- **Features**: 13 clinical attributes (age, chest pain, cholesterol, etc.)
- **Target**: Binary classification (heart disease presence/absence)
- **Accuracy**: Up to 87% with optimized models

### 3. Echocardiogram Dataset
- **Samples**: 132 patients post-heart attack
- **Features**: 13 echocardiogram measurements
- **Target**: 1-year survival prediction
- **Challenge**: Handling missing data and class imbalance

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Hugging Face token
```

### Running the Analysis

```bash
# Heart Disease Prediction
python src/heart_disease_predictor.py

# Echocardiogram Survival Analysis
python src/echocardiogram_analyzer.py

# Medical LLM Fine-tuning (requires GPU)
python src/medical_llm_trainer.py
```

## 📁 Project Structure

```
medical-ai-training/
├── src/                          # Source code
│   ├── medical_llm_trainer.py    # LLM fine-tuning pipeline
│   ├── heart_disease_predictor.py # Heart disease ML models
│   └── echocardiogram_analyzer.py # Echocardiogram analysis
├── data/                         # Dataset directory
│   ├── statlog+heart/           # Heart disease data
│   ├── echocardiogram/          # Echocardiogram data
│   └── KDIGO-2022-Clinical-Practice-Guideline-for-Diabetes-Management-in-CKD.pdf
├── results/                      # Model outputs & visualizations
├── notebooks/                    # Jupyter notebooks (optional)
├── docs/                        # Additional documentation
├── config.py                    # Configuration management
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔧 Features

### Medical LLM Fine-tuning
- **Model**: Llama-2-13b-chat-hf
- **Technique**: Supervised fine-tuning on clinical guidelines
- **Features**: 
  - PDF text extraction and preprocessing
  - Chunked training data preparation
  - GPU-optimized training pipeline
  - Model validation and saving

### Heart Disease Prediction
- **Models**: Random Forest, Gradient Boosting, SVM, Logistic Regression
- **Features**:
  - Comprehensive EDA with 12+ visualizations
  - Feature importance analysis
  - Cross-validation and hyperparameter tuning
  - ROC/AUC analysis and clinical insights

### Echocardiogram Analysis
- **Challenge**: Survival prediction with missing data
- **Features**:
  - Advanced missing data imputation
  - Clinical feature correlation analysis
  - Survival rate analysis by patient groups
  - Model interpretability for clinical use

## 📈 Results

### Heart Disease Prediction
- **Best Model**: Random Forest
- **Accuracy**: 87%
- **AUC Score**: 0.92
- **Key Features**: Chest pain type, exercise angina, ST depression

### Echocardiogram Analysis
- **Best Model**: Gradient Boosting
- **Accuracy**: 78%
- **AUC Score**: 0.85
- **Key Features**: Wall motion index, fractional shortening, age

### LLM Fine-tuning
- **Model Size**: 13B parameters
- **Training**: Diabetes management guidelines
- **Application**: Medical Q&A system

## 🔬 Technical Details

### Machine Learning Pipeline
1. **Data Loading & Validation**
2. **Exploratory Data Analysis**
3. **Feature Engineering & Preprocessing**
4. **Model Training & Cross-validation**
5. **Evaluation & Clinical Insights**
6. **Model Persistence & Deployment**

### LLM Fine-tuning Pipeline
1. **PDF Text Extraction**
2. **Text Chunking & Tokenization**
3. **Dataset Preparation**
4. **Distributed Training Setup**
5. **Model Fine-tuning**
6. **Validation & Saving**

## 📊 Visualizations

The system generates comprehensive visualizations:

- **EDA Plots**: Distribution analysis, correlation heatmaps
- **Model Performance**: ROC curves, precision-recall curves
- **Feature Importance**: Clinical relevance analysis
- **Survival Analysis**: Kaplan-Meier style insights

## 🏥 Clinical Applications

### Heart Disease Screening
- **Use Case**: Early detection in primary care
- **Impact**: Reduce cardiovascular events
- **Integration**: EMR systems, clinical decision support

### Post-MI Survival Prediction
- **Use Case**: Risk stratification after heart attacks
- **Impact**: Personalized treatment planning
- **Integration**: Cardiology departments, ICU monitoring

### Medical AI Assistant
- **Use Case**: Clinical guideline consultation
- **Impact**: Evidence-based decision making
- **Integration**: Medical education, clinical practice

## 🛠️ Configuration

### Environment Variables
```bash
# Required
HUGGINGFACE_TOKEN=your_token_here

# Optional (with defaults)
MODEL_NAME=meta-llama/Llama-2-13b-chat-hf
BATCH_SIZE=1
MAX_LENGTH=512
NUM_EPOCHS=1
```

### Hardware Requirements
- **LLM Training**: GPU with 16GB+ VRAM
- **ML Models**: CPU sufficient, GPU recommended
- **Memory**: 8GB+ RAM recommended

## 📚 Dependencies

### Core Libraries
- `transformers` - Hugging Face transformers
- `torch` - PyTorch deep learning framework
- `scikit-learn` - Machine learning algorithms
- `pandas` - Data manipulation
- `numpy` - Numerical computing

### Visualization
- `matplotlib` - Plotting library
- `seaborn` - Statistical visualization
- `plotly` - Interactive plots (optional)

### Medical/Text Processing
- `PyPDF2` - PDF text extraction
- `python-dotenv` - Environment management
- `joblib` - Model persistence

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Rutgers Health Hack 2024** - Hackathon organizers
- **UCI ML Repository** - Heart disease and echocardiogram datasets
- **KDIGO** - Clinical practice guidelines
- **Hugging Face** - Model hosting and transformers library

## 📞 Contact

**Healthcare AI Team**
- Email: [your-email@example.com]
- LinkedIn: [Your LinkedIn Profile]
- GitHub: [Your GitHub Profile]

---

*Developed with ❤️ for advancing healthcare through AI* 