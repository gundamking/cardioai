"""
Heart Disease Prediction Analysis
=================================

This module provides comprehensive analysis and prediction capabilities for heart disease
using the Statlog Heart dataset. It includes data preprocessing, exploratory data analysis,
model training, and evaluation.

Dataset: Statlog Heart Disease Dataset
Features: 13 clinical attributes
Target: Binary classification (presence/absence of heart disease)

Author: Healthcare AI Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    precision_recall_curve,
    accuracy_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class HeartDiseasePredictor:
    """
    Comprehensive heart disease prediction system
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the predictor with data path
        
        Args:
            data_path (str): Path to the heart disease dataset
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
        # Feature names for the Statlog Heart dataset
        self.feature_names = [
            'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
            'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate',
            'exercise_angina', 'st_depression', 'st_slope', 'vessels_colored',
            'thalassemia'
        ]
        
        # Feature descriptions
        self.feature_descriptions = {
            'age': 'Age in years',
            'sex': 'Sex (1 = male, 0 = female)',
            'chest_pain_type': 'Chest pain type (1-4)',
            'resting_bp': 'Resting blood pressure (mm Hg)',
            'cholesterol': 'Serum cholesterol (mg/dl)',
            'fasting_blood_sugar': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
            'resting_ecg': 'Resting electrocardiographic results (0-2)',
            'max_heart_rate': 'Maximum heart rate achieved',
            'exercise_angina': 'Exercise induced angina (1 = yes, 0 = no)',
            'st_depression': 'ST depression induced by exercise relative to rest',
            'st_slope': 'Slope of the peak exercise ST segment (1-3)',
            'vessels_colored': 'Number of major vessels colored by fluoroscopy (0-3)',
            'thalassemia': 'Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)'
        }
    
    def load_data(self):
        """
        Load and prepare the heart disease dataset
        """
        try:
            # Load data (no header in the original file)
            self.data = pd.read_csv(self.data_path, sep=' ', header=None)
            self.data.columns = self.feature_names + ['target']
            
            # Convert target to binary (1 = disease, 0 = no disease)
            self.data['target'] = (self.data['target'] == 2).astype(int)
            
            print(f"✓ Data loaded successfully: {self.data.shape[0]} samples, {self.data.shape[1]-1} features")
            print(f"✓ Target distribution: {self.data['target'].value_counts().to_dict()}")
            
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            raise
    
    def explore_data(self):
        """
        Perform comprehensive exploratory data analysis
        """
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic statistics
        print("\n1. Dataset Overview:")
        print(f"   Shape: {self.data.shape}")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Missing values: {self.data.isnull().sum().sum()}")
        
        # Target distribution
        print("\n2. Target Distribution:")
        target_counts = self.data['target'].value_counts()
        print(f"   No Disease (0): {target_counts[0]} ({target_counts[0]/len(self.data)*100:.1f}%)")
        print(f"   Disease (1): {target_counts[1]} ({target_counts[1]/len(self.data)*100:.1f}%)")
        
        # Feature statistics
        print("\n3. Feature Statistics:")
        print(self.data.describe())
        
        # Create visualizations
        self._create_visualizations()
    
    def _create_visualizations(self):
        """
        Create comprehensive visualizations for EDA
        """
        # Set up the plotting area
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Target distribution
        plt.subplot(3, 4, 1)
        self.data['target'].value_counts().plot(kind='bar', color=['lightblue', 'lightcoral'])
        plt.title('Target Distribution')
        plt.xlabel('Heart Disease')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['No Disease', 'Disease'], rotation=0)
        
        # 2. Age distribution by target
        plt.subplot(3, 4, 2)
        self.data.boxplot(column='age', by='target', ax=plt.gca())
        plt.title('Age Distribution by Heart Disease')
        plt.suptitle('')
        
        # 3. Chest pain type distribution
        plt.subplot(3, 4, 3)
        chest_pain_counts = self.data.groupby(['chest_pain_type', 'target']).size().unstack()
        chest_pain_counts.plot(kind='bar', ax=plt.gca())
        plt.title('Chest Pain Type vs Heart Disease')
        plt.legend(['No Disease', 'Disease'])
        plt.xticks(rotation=0)
        
        # 4. Cholesterol distribution
        plt.subplot(3, 4, 4)
        self.data.boxplot(column='cholesterol', by='target', ax=plt.gca())
        plt.title('Cholesterol by Heart Disease')
        plt.suptitle('')
        
        # 5. Max heart rate distribution
        plt.subplot(3, 4, 5)
        self.data.boxplot(column='max_heart_rate', by='target', ax=plt.gca())
        plt.title('Max Heart Rate by Heart Disease')
        plt.suptitle('')
        
        # 6. Exercise angina
        plt.subplot(3, 4, 6)
        angina_counts = self.data.groupby(['exercise_angina', 'target']).size().unstack()
        angina_counts.plot(kind='bar', ax=plt.gca())
        plt.title('Exercise Angina vs Heart Disease')
        plt.legend(['No Disease', 'Disease'])
        plt.xticks([0, 1], ['No Angina', 'Angina'], rotation=0)
        
        # 7. ST Depression
        plt.subplot(3, 4, 7)
        self.data.boxplot(column='st_depression', by='target', ax=plt.gca())
        plt.title('ST Depression by Heart Disease')
        plt.suptitle('')
        
        # 8. Correlation heatmap
        plt.subplot(3, 4, 8)
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=plt.gca(), cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix')
        
        # 9. Sex distribution
        plt.subplot(3, 4, 9)
        sex_counts = self.data.groupby(['sex', 'target']).size().unstack()
        sex_counts.plot(kind='bar', ax=plt.gca())
        plt.title('Sex vs Heart Disease')
        plt.legend(['No Disease', 'Disease'])
        plt.xticks([0, 1], ['Female', 'Male'], rotation=0)
        
        # 10. Thalassemia distribution
        plt.subplot(3, 4, 10)
        thal_counts = self.data.groupby(['thalassemia', 'target']).size().unstack()
        thal_counts.plot(kind='bar', ax=plt.gca())
        plt.title('Thalassemia vs Heart Disease')
        plt.legend(['No Disease', 'Disease'])
        plt.xticks(rotation=0)
        
        # 11. Vessels colored
        plt.subplot(3, 4, 11)
        vessels_counts = self.data.groupby(['vessels_colored', 'target']).size().unstack()
        vessels_counts.plot(kind='bar', ax=plt.gca())
        plt.title('Vessels Colored vs Heart Disease')
        plt.legend(['No Disease', 'Disease'])
        plt.xticks(rotation=0)
        
        # 12. Feature importance preview (using random forest)
        plt.subplot(3, 4, 12)
        X_temp = self.data.drop('target', axis=1)
        y_temp = self.data['target']
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_temp.fit(X_temp, y_temp)
        feature_importance = pd.DataFrame({
            'feature': X_temp.columns,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.barh(range(len(feature_importance)), feature_importance['importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.title('Feature Importance (Random Forest)')
        plt.xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('results/heart_disease_eda.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_features(self):
        """
        Prepare features for machine learning
        """
        # Separate features and target
        self.X = self.data.drop('target', axis=1)
        self.y = self.data['target']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✓ Features prepared: {self.X_train.shape[0]} training, {self.X_test.shape[0]} testing samples")
    
    def train_models(self):
        """
        Train multiple machine learning models
        """
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for SVM and Logistic Regression
            if name in ['SVM', 'Logistic Regression']:
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test
            
            # Train model
            model.fit(X_train_use, self.y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_use)
            y_pred_proba = model.predict_proba(X_test_use)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_use, self.y_train, cv=5, scoring='accuracy')
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'auc': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUC: {auc_score:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    def evaluate_models(self):
        """
        Comprehensive model evaluation
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Create evaluation plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model comparison
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        auc_scores = [self.results[name]['auc'] for name in model_names]
        
        ax1 = axes[0, 0]
        x = np.arange(len(model_names))
        width = 0.35
        ax1.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        ax1.bar(x + width/2, auc_scores, width, label='AUC', alpha=0.8)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. ROC Curves
        ax2 = axes[0, 1]
        for name in model_names:
            y_pred_proba = self.results[name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = self.results[name]['auc']
            ax2.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curves
        ax3 = axes[1, 0]
        for name in model_names:
            y_pred_proba = self.results[name]['y_pred_proba']
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            ax3.plot(recall, precision, label=name)
        
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curves')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature importance (best model)
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc'])
        best_model = self.models[best_model_name]
        
        ax4 = axes[1, 1]
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            ax4.barh(range(len(feature_importance)), feature_importance['importance'])
            ax4.set_yticks(range(len(feature_importance)))
            ax4.set_yticklabels(feature_importance['feature'])
            ax4.set_xlabel('Importance')
            ax4.set_title(f'Feature Importance ({best_model_name})')
        
        plt.tight_layout()
        plt.savefig('results/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed results
        print(f"\nBest Model: {best_model_name}")
        print(f"Best AUC Score: {self.results[best_model_name]['auc']:.4f}")
        
        # Detailed classification report for best model
        print(f"\nClassification Report ({best_model_name}):")
        print(classification_report(self.y_test, self.results[best_model_name]['y_pred']))
    
    def save_model(self, model_name: str = None):
        """
        Save the best model
        
        Args:
            model_name (str): Name of the model to save. If None, saves the best model.
        """
        if model_name is None:
            model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc'])
        
        model_path = f'results/heart_disease_model_{model_name.lower().replace(" ", "_")}.joblib'
        scaler_path = f'results/heart_disease_scaler.joblib'
        
        joblib.dump(self.models[model_name], model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"✓ Model saved: {model_path}")
        print(f"✓ Scaler saved: {scaler_path}")
    
    def run_complete_analysis(self):
        """
        Run the complete heart disease prediction analysis
        """
        print("🫀 HEART DISEASE PREDICTION ANALYSIS")
        print("="*60)
        
        # Load and explore data
        self.load_data()
        self.explore_data()
        
        # Prepare features
        self.prepare_features()
        
        # Train models
        self.train_models()
        
        # Evaluate models
        self.evaluate_models()
        
        # Save best model
        self.save_model()
        
        print("\n✅ Analysis completed successfully!")
        print("📊 Results saved in 'results/' directory")


def main():
    """
    Main function to run heart disease prediction analysis
    """
    # Initialize predictor
    predictor = HeartDiseasePredictor('../data/statlog+heart/heart.dat')
    
    # Run complete analysis
    predictor.run_complete_analysis()


if __name__ == "__main__":
    main() 