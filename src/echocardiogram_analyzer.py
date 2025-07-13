"""
Echocardiogram Survival Prediction Analysis
==========================================

This module provides comprehensive analysis for predicting patient survival
after heart attacks using echocardiogram data. The analysis focuses on
identifying patients who will survive at least one year post-heart attack.

Dataset: Echocardiogram Dataset (132 patients)
Features: 13 clinical measurements from echocardiograms
Target: Survival prediction (alive at 1 year)

Author: Healthcare AI Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
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
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EchocardiogramAnalyzer:
    """
    Comprehensive echocardiogram survival prediction system
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the analyzer with data path
        
        Args:
            data_path (str): Path to the echocardiogram dataset
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
        self.imputer = SimpleImputer(strategy='median')
        self.models = {}
        self.results = {}
        
        # Feature names based on the dataset documentation
        self.feature_names = [
            'survival_months', 'still_alive', 'age_at_attack', 'pericardial_effusion',
            'fractional_shortening', 'epss', 'lvdd', 'wall_motion_score',
            'wall_motion_index', 'mult', 'name', 'group', 'alive_at_1'
        ]
        
        # Features to use for prediction (excluding target and irrelevant features)
        self.prediction_features = [
            'age_at_attack', 'pericardial_effusion', 'fractional_shortening',
            'epss', 'lvdd', 'wall_motion_score', 'wall_motion_index'
        ]
        
        # Feature descriptions
        self.feature_descriptions = {
            'survival_months': 'Number of months patient survived',
            'still_alive': 'Binary: 0=dead, 1=still alive',
            'age_at_attack': 'Age when heart attack occurred',
            'pericardial_effusion': 'Fluid around heart: 0=no, 1=yes',
            'fractional_shortening': 'Heart contractility measure (lower = worse)',
            'epss': 'E-point septal separation (higher = worse)',
            'lvdd': 'Left ventricular end-diastolic dimension (larger = worse)',
            'wall_motion_score': 'How heart segments move',
            'wall_motion_index': 'Wall motion score / segments seen',
            'alive_at_1': 'Alive at 1 year: 0=no, 1=yes'
        }
    
    def load_data(self):
        """
        Load and prepare the echocardiogram dataset
        """
        try:
            # Load data
            self.data = pd.read_csv(self.data_path, sep=',', header=None)
            self.data.columns = self.feature_names
            
            # Replace '?' with NaN
            self.data = self.data.replace('?', np.nan)
            
            # Convert numeric columns
            numeric_columns = [col for col in self.data.columns if col not in ['name']]
            for col in numeric_columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # Remove rows where target is missing
            self.data = self.data.dropna(subset=['alive_at_1'])
            
            print(f"✓ Data loaded successfully: {self.data.shape[0]} samples, {self.data.shape[1]} features")
            print(f"✓ Target distribution: {self.data['alive_at_1'].value_counts().to_dict()}")
            print(f"✓ Missing values per feature:")
            missing_counts = self.data[self.prediction_features + ['alive_at_1']].isnull().sum()
            for feature, count in missing_counts.items():
                if count > 0:
                    print(f"   {feature}: {count} ({count/len(self.data)*100:.1f}%)")
            
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
        print(f"   Prediction features: {len(self.prediction_features)}")
        print(f"   Total missing values: {self.data.isnull().sum().sum()}")
        
        # Target distribution
        print("\n2. Survival at 1 Year:")
        target_counts = self.data['alive_at_1'].value_counts()
        print(f"   Died (0): {target_counts[0]} ({target_counts[0]/len(self.data)*100:.1f}%)")
        print(f"   Survived (1): {target_counts[1]} ({target_counts[1]/len(self.data)*100:.1f}%)")
        
        # Feature statistics
        print("\n3. Feature Statistics:")
        print(self.data[self.prediction_features].describe())
        
        # Create visualizations
        self._create_visualizations()
    
    def _create_visualizations(self):
        """
        Create comprehensive visualizations for EDA
        """
        # Set up the plotting area
        fig = plt.figure(figsize=(18, 14))
        
        # 1. Target distribution
        plt.subplot(3, 4, 1)
        self.data['alive_at_1'].value_counts().plot(kind='bar', color=['lightcoral', 'lightblue'])
        plt.title('Survival at 1 Year')
        plt.xlabel('Outcome')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Died', 'Survived'], rotation=0)
        
        # 2. Age distribution by survival
        plt.subplot(3, 4, 2)
        self.data.boxplot(column='age_at_attack', by='alive_at_1', ax=plt.gca())
        plt.title('Age at Attack by Survival')
        plt.suptitle('')
        
        # 3. Pericardial effusion
        plt.subplot(3, 4, 3)
        if not self.data['pericardial_effusion'].isnull().all():
            effusion_counts = self.data.groupby(['pericardial_effusion', 'alive_at_1']).size().unstack(fill_value=0)
            effusion_counts.plot(kind='bar', ax=plt.gca())
            plt.title('Pericardial Effusion vs Survival')
            plt.legend(['Died', 'Survived'])
            plt.xticks([0, 1], ['No Fluid', 'Fluid'], rotation=0)
        
        # 4. Fractional shortening
        plt.subplot(3, 4, 4)
        self.data.boxplot(column='fractional_shortening', by='alive_at_1', ax=plt.gca())
        plt.title('Fractional Shortening by Survival')
        plt.suptitle('')
        
        # 5. EPSS (E-point septal separation)
        plt.subplot(3, 4, 5)
        self.data.boxplot(column='epss', by='alive_at_1', ax=plt.gca())
        plt.title('EPSS by Survival')
        plt.suptitle('')
        
        # 6. LVDD (Left ventricular dimension)
        plt.subplot(3, 4, 6)
        self.data.boxplot(column='lvdd', by='alive_at_1', ax=plt.gca())
        plt.title('LVDD by Survival')
        plt.suptitle('')
        
        # 7. Wall motion score
        plt.subplot(3, 4, 7)
        self.data.boxplot(column='wall_motion_score', by='alive_at_1', ax=plt.gca())
        plt.title('Wall Motion Score by Survival')
        plt.suptitle('')
        
        # 8. Wall motion index
        plt.subplot(3, 4, 8)
        self.data.boxplot(column='wall_motion_index', by='alive_at_1', ax=plt.gca())
        plt.title('Wall Motion Index by Survival')
        plt.suptitle('')
        
        # 9. Correlation heatmap
        plt.subplot(3, 4, 9)
        correlation_data = self.data[self.prediction_features + ['alive_at_1']].corr()
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=plt.gca(), cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix')
        
        # 10. Survival months distribution
        plt.subplot(3, 4, 10)
        self.data['survival_months'].hist(bins=20, alpha=0.7, ax=plt.gca())
        plt.title('Survival Months Distribution')
        plt.xlabel('Months')
        plt.ylabel('Frequency')
        
        # 11. Still alive vs survived at 1 year
        plt.subplot(3, 4, 11)
        if not self.data['still_alive'].isnull().all():
            alive_counts = self.data.groupby(['still_alive', 'alive_at_1']).size().unstack(fill_value=0)
            alive_counts.plot(kind='bar', ax=plt.gca())
            plt.title('Still Alive vs 1-Year Survival')
            plt.legend(['Died', 'Survived'])
            plt.xticks([0, 1], ['Dead', 'Alive'], rotation=0)
        
        # 12. Missing data pattern
        plt.subplot(3, 4, 12)
        missing_data = self.data[self.prediction_features].isnull().sum()
        missing_data.plot(kind='bar', ax=plt.gca())
        plt.title('Missing Data by Feature')
        plt.xlabel('Features')
        plt.ylabel('Missing Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/echocardiogram_eda.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_features(self):
        """
        Prepare features for machine learning
        """
        # Select features and target
        self.X = self.data[self.prediction_features].copy()
        self.y = self.data['alive_at_1'].copy()
        
        # Handle missing values
        self.X_imputed = self.imputer.fit_transform(self.X)
        self.X = pd.DataFrame(self.X_imputed, columns=self.prediction_features)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✓ Features prepared: {self.X_train.shape[0]} training, {self.X_test.shape[0]} testing samples")
        print(f"✓ Missing values handled using median imputation")
    
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
                'feature': self.prediction_features,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            ax4.barh(range(len(feature_importance)), feature_importance['importance'])
            ax4.set_yticks(range(len(feature_importance)))
            ax4.set_yticklabels(feature_importance['feature'])
            ax4.set_xlabel('Importance')
            ax4.set_title(f'Feature Importance ({best_model_name})')
        
        plt.tight_layout()
        plt.savefig('results/echocardiogram_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed results
        print(f"\nBest Model: {best_model_name}")
        print(f"Best AUC Score: {self.results[best_model_name]['auc']:.4f}")
        
        # Detailed classification report for best model
        print(f"\nClassification Report ({best_model_name}):")
        print(classification_report(self.y_test, self.results[best_model_name]['y_pred']))
        
        # Clinical insights
        self._generate_clinical_insights()
    
    def _generate_clinical_insights(self):
        """
        Generate clinical insights from the analysis
        """
        print("\n" + "="*50)
        print("CLINICAL INSIGHTS")
        print("="*50)
        
        # Feature correlations with survival
        correlations = self.data[self.prediction_features + ['alive_at_1']].corr()['alive_at_1'].drop('alive_at_1')
        
        print("\nFeature Correlations with 1-Year Survival:")
        for feature, corr in correlations.sort_values(key=abs, ascending=False).items():
            direction = "positively" if corr > 0 else "negatively"
            strength = "strongly" if abs(corr) > 0.3 else "moderately" if abs(corr) > 0.1 else "weakly"
            print(f"  {feature}: {corr:.3f} ({strength} {direction} correlated)")
        
        # Survival statistics by key features
        print("\nSurvival Rates by Key Clinical Indicators:")
        
        # Age groups
        age_groups = pd.cut(self.data['age_at_attack'], bins=[0, 50, 60, 70, 100], 
                           labels=['<50', '50-60', '60-70', '>70'])
        age_survival = self.data.groupby(age_groups)['alive_at_1'].agg(['count', 'mean'])
        print("\nAge Groups:")
        for age_group, stats in age_survival.iterrows():
            print(f"  {age_group}: {stats['mean']:.1%} survival ({stats['count']} patients)")
        
        # Pericardial effusion
        if not self.data['pericardial_effusion'].isnull().all():
            effusion_survival = self.data.groupby('pericardial_effusion')['alive_at_1'].agg(['count', 'mean'])
            print("\nPericardial Effusion:")
            for effusion, stats in effusion_survival.iterrows():
                status = "No fluid" if effusion == 0 else "Fluid present"
                print(f"  {status}: {stats['mean']:.1%} survival ({stats['count']} patients)")
    
    def save_model(self, model_name: str = None):
        """
        Save the best model
        
        Args:
            model_name (str): Name of the model to save. If None, saves the best model.
        """
        if model_name is None:
            model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc'])
        
        model_path = f'results/echocardiogram_model_{model_name.lower().replace(" ", "_")}.joblib'
        scaler_path = f'results/echocardiogram_scaler.joblib'
        imputer_path = f'results/echocardiogram_imputer.joblib'
        
        joblib.dump(self.models[model_name], model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.imputer, imputer_path)
        
        print(f"✓ Model saved: {model_path}")
        print(f"✓ Scaler saved: {scaler_path}")
        print(f"✓ Imputer saved: {imputer_path}")
    
    def run_complete_analysis(self):
        """
        Run the complete echocardiogram survival analysis
        """
        print("🫀 ECHOCARDIOGRAM SURVIVAL PREDICTION ANALYSIS")
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
    Main function to run echocardiogram survival analysis
    """
    # Initialize analyzer
    analyzer = EchocardiogramAnalyzer('../data/echocardiogram/echocardiogram.data')
    
    # Run complete analysis
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main() 