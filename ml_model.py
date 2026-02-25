import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, log_loss
from sklearn.calibration import CalibratedClassifierCV
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MLBasketballPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_cols = None
        self.feature_importances = None
    
    def load_data(self, filepath='data/ml_ready_data.csv'):
        """Load the prepared dataset"""
        print(f"Loading data from {filepath}...")
        self.df = pd.read_csv(filepath)
        
        # Load feature columns
        with open('data/feature_columns.json', 'r') as f:
            self.feature_cols = json.load(f)
        
        print(f"✅ Loaded {len(self.df)} games with {len(self.feature_cols)} features")
        return self.df
    
    def prepare_data(self, test_size=0.2, scale=True):
        """Prepare train/test split with optional scaling and imputation"""
        # Sort by date for temporal split
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')
        
        # Split chronologically
        split_idx = int(len(self.df) * (1 - test_size))
        
        X = self.df[self.feature_cols]
        y = self.df['home_win']
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]
        
        # Impute missing values
        self.X_train = pd.DataFrame(
            self.imputer.fit_transform(X_train),
            columns=self.feature_cols,
            index=X_train.index
        )
        self.X_test = pd.DataFrame(
            self.imputer.transform(X_test),
            columns=self.feature_cols,
            index=X_test.index
        )
        
        if scale:
            self.X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(self.X_train),
                columns=self.feature_cols,
                index=self.X_train.index
            )
            self.X_test_scaled = pd.DataFrame(
                self.scaler.transform(self.X_test),
                columns=self.feature_cols,
                index=self.X_test.index
            )
        else:
            self.X_train_scaled = self.X_train
            self.X_test_scaled = self.X_test
        
        print(f"\n📊 Data Split:")
        print(f"  Training: {len(self.X_train)} games ({self.y_train.mean():.1%} home wins)")
        print(f"  Testing: {len(self.X_test)} games ({self.y_test.mean():.1%} home wins)")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_random_forest(self, tune_hyperparameters=False):
        """Train Random Forest with optional hyperparameter tuning"""
        print("\n🌲 Training Random Forest...")
        
        if tune_hyperparameters:
            print("  Tuning hyperparameters (this may take a while)...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [20, 50, 100],
                'min_samples_leaf': [10, 20, 30],
                'max_features': ['sqrt', 'log2']
            }
            
            rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
            grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"  Best parameters: {grid_search.best_params_}")
            model = grid_search.best_estimator_
        else:
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=30,
                min_samples_leaf=15,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            model.fit(self.X_train, self.y_train)
        
        return model
    
    def train_gradient_boosting(self, tune_hyperparameters=False):
        """Train Histogram Gradient Boosting (handles NaN natively)"""
        print("\n🚀 Training Gradient Boosting...")
        
        if tune_hyperparameters:
            print("  Tuning hyperparameters...")
            param_grid = {
                'max_iter': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'max_leaf_nodes': [15, 31, 63]
            }
            
            gb = HistGradientBoostingClassifier(random_state=42)
            grid_search = GridSearchCV(gb, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"  Best parameters: {grid_search.best_params_}")
            model = grid_search.best_estimator_
        else:
            model = HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_depth=5,
                max_leaf_nodes=31,
                random_state=42
            )
            model.fit(self.X_train, self.y_train)
        
        return model
    
    def train_logistic_regression(self):
        """Train Logistic Regression (needs scaled data)"""
        print("\n📈 Training Logistic Regression...")
        
        model = LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight='balanced',
            C=0.1
        )
        model.fit(self.X_train_scaled, self.y_train)
        
        return model
    
    def train_ensemble(self):
        """Train ensemble of multiple models"""
        print("\n🎯 Training Ensemble Model...")
        
        # Train individual models
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=30,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        gb = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        
        lr = LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight='balanced',
            C=0.1
        )
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('lr', lr)
            ],
            voting='soft',
            weights=[2, 2, 1]
        )
        
        ensemble.fit(self.X_train, self.y_train)
        
        return ensemble
    
    def evaluate_model(self, model, model_name, X_train, X_test):
        """Comprehensive model evaluation"""
        print(f"\n{'='*60}")
        print(f"📊 {model_name} Performance")
        print(f"{'='*60}")
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        
        # Accuracies
        train_acc = accuracy_score(self.y_train, train_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        
        print(f"\n🎯 Accuracy:")
        print(f"  Training: {train_acc:.4f}")
        print(f"  Testing:  {test_acc:.4f}")
        print(f"  Gap:      {train_acc - test_acc:.4f}")
        
        # AUC
        train_auc = roc_auc_score(self.y_train, train_proba)
        test_auc = roc_auc_score(self.y_test, test_proba)
        
        print(f"\n📈 AUC-ROC:")
        print(f"  Training: {train_auc:.4f}")
        print(f"  Testing:  {test_auc:.4f}")
        
        # Log Loss
        train_logloss = log_loss(self.y_train, train_proba)
        test_logloss = log_loss(self.y_test, test_proba)
        
        print(f"\n📉 Log Loss (lower is better):")
        print(f"  Training: {train_logloss:.4f}")
        print(f"  Testing:  {test_logloss:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, test_pred)
        
        print(f"\n📊 Confusion Matrix (Test Set):")
        print(f"                Predicted Away  Predicted Home")
        print(f"  Actual Away:  {cm[0,0]:6d}          {cm[0,1]:6d}")
        print(f"  Actual Home:  {cm[1,0]:6d}          {cm[1,1]:6d}")
        
        # Specific metrics
        away_precision = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
        away_recall = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        home_precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        home_recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        
        print(f"\n🎲 Detailed Metrics:")
        print(f"  Away Team Precision: {away_precision:.3f} (when model predicts away, how often correct)")
        print(f"  Away Team Recall:    {away_recall:.3f} (of actual away wins, how many caught)")
        print(f"  Home Team Precision: {home_precision:.3f}")
        print(f"  Home Team Recall:    {home_recall:.3f}")
        
        # Feature Importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔝 Top 15 Most Important Features:")
            for idx, row in importances.head(15).iterrows():
                print(f"    {row['feature']:40s}: {row['importance']:.4f}")
            
            self.feature_importances = importances
        
        # Probability calibration
        bins = [0, 0.4, 0.5, 0.6, 1.0]
        bin_labels = ['<40%', '40-50%', '50-60%', '>60%']
        
        test_df = pd.DataFrame({
            'actual': self.y_test.values,
            'predicted_prob': test_proba
        })
        test_df['bin'] = pd.cut(test_df['predicted_prob'], bins=bins, labels=bin_labels)
        
        print(f"\n🎯 Probability Calibration:")
        for bin_name in bin_labels:
            bin_data = test_df[test_df['bin'] == bin_name]
            if len(bin_data) > 0:
                actual_rate = bin_data['actual'].mean()
                avg_pred = bin_data['predicted_prob'].mean()
                print(f"  {bin_name} predicted: {avg_pred:.3f} actual: {actual_rate:.3f} (n={len(bin_data)})")
        
        return {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'test_logloss': test_logloss
        }
    
    def compare_all_models(self):
        """Train and compare all model types"""
        print("\n" + "="*60)
        print("🔬 COMPARING ALL MODELS")
        print("="*60)
        
        self.prepare_data(scale=True)
        
        results = {}
        
        # Random Forest
        rf_model = self.train_random_forest()
        results['Random Forest'] = self.evaluate_model(rf_model, 'Random Forest', 
                                                       self.X_train, self.X_test)
        
        # Gradient Boosting
        gb_model = self.train_gradient_boosting()
        results['Gradient Boosting'] = self.evaluate_model(gb_model, 'Gradient Boosting',
                                                           self.X_train, self.X_test)
        
        # Logistic Regression
        lr_model = self.train_logistic_regression()
        results['Logistic Regression'] = self.evaluate_model(lr_model, 'Logistic Regression',
                                                             self.X_train_scaled, self.X_test_scaled)
        
        # Ensemble
        ensemble_model = self.train_ensemble()
        results['Ensemble'] = self.evaluate_model(ensemble_model, 'Ensemble',
                                                  self.X_train, self.X_test)
        
        # Summary comparison
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Model':<25} {'Test Acc':<12} {'Test AUC':<12} {'Log Loss':<12}")
        print("-"*60)
        
        for model_name, metrics in results.items():
            print(f"{model_name:<25} {metrics['test_acc']:<12.4f} {metrics['test_auc']:<12.4f} {metrics['test_logloss']:<12.4f}")
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['test_auc'])
        print(f"\n🏆 Best Model: {best_model_name} (AUC: {results[best_model_name]['test_auc']:.4f})")
        
        # Save best model
        if best_model_name == 'Random Forest':
            best_model = rf_model
        elif best_model_name == 'Gradient Boosting':
            best_model = gb_model
        elif best_model_name == 'Logistic Regression':
            best_model = lr_model
        else:
            best_model = ensemble_model
        
        return best_model, results
    
    def save_model(self, model, model_name='best_model'):
        """Save model and preprocessing objects"""
        import os
        os.makedirs('models', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_file = f"models/{model_name}_{timestamp}.pkl"
        joblib.dump(model, model_file)
        
        # Save scaler
        scaler_file = f"models/scaler_{timestamp}.pkl"
        joblib.dump(self.scaler, scaler_file)
        
        # Save imputer
        imputer_file = f"models/imputer_{timestamp}.pkl"
        joblib.dump(self.imputer, imputer_file)
        
        # Save feature columns
        features_file = f"models/features_{timestamp}.json"
        with open(features_file, 'w') as f:
            json.dump(self.feature_cols, f)
        
        print(f"\n💾 Model saved:")
        print(f"  Model: {model_file}")
        print(f"  Scaler: {scaler_file}")
        print(f"  Imputer: {imputer_file}")
        print(f"  Features: {features_file}")
        
        return model_file

# Usage
if __name__ == "__main__":
    predictor = MLBasketballPredictor()
    predictor.load_data()
    
    # Compare all models
    best_model, results = predictor.compare_all_models()
    
    # Save the best model
    predictor.save_model(best_model, 'best_ensemble_model')
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)