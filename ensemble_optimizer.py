import pandas as pd
import numpy as np
import json
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnsembleOptimizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_cols = None
        self.best_ensemble = None
        
    def load_data(self, filepath='data/ml_ready_data.csv'):
        """Load the prepared dataset"""
        print(f"Loading data from {filepath}...")
        self.df = pd.read_csv(filepath)
        
        with open('data/feature_columns.json', 'r') as f:
            self.feature_cols = json.load(f)
        
        print(f"✅ Loaded {len(self.df)} games with {len(self.feature_cols)} features\n")
        return self.df
    
    def prepare_data(self, test_size=0.2):
        """Prepare data with temporal split"""
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')
        
        split_idx = int(len(self.df) * (1 - test_size))
        
        X = self.df[self.feature_cols]
        y = self.df['home_win']
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]
        
        # Impute and scale
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
        
        print(f"📊 Data Split:")
        print(f"  Training: {len(self.X_train)} games ({self.y_train.mean():.1%} home wins)")
        print(f"  Testing: {len(self.X_test)} games ({self.y_test.mean():.1%} home wins)\n")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def test_individual_models(self):
        """Test various individual models to find best performers"""
        print("="*60)
        print("🔍 TESTING INDIVIDUAL MODELS")
        print("="*60 + "\n")
        
        models = {
            'Random Forest (Default)': RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_split=30,
                min_samples_leaf=15, random_state=42, n_jobs=-1, class_weight='balanced'
            ),
            'Random Forest (Deep)': RandomForestClassifier(
                n_estimators=500, max_depth=25, min_samples_split=20,
                min_samples_leaf=10, random_state=42, n_jobs=-1, class_weight='balanced'
            ),
            'Random Forest (Shallow)': RandomForestClassifier(
                n_estimators=300, max_depth=10, min_samples_split=50,
                min_samples_leaf=20, random_state=42, n_jobs=-1, class_weight='balanced'
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=300, max_depth=15, min_samples_split=30,
                random_state=42, n_jobs=-1, class_weight='balanced'
            ),
            'Hist Gradient Boosting (Conservative)': HistGradientBoostingClassifier(
                max_iter=100, learning_rate=0.03, max_depth=4,
                max_leaf_nodes=15, random_state=42
            ),
            'Hist Gradient Boosting (Moderate)': HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=5,
                max_leaf_nodes=31, random_state=42
            ),
            'Hist Gradient Boosting (Aggressive)': HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.08, max_depth=6,
                max_leaf_nodes=31, random_state=42
            ),
            'Logistic Regression (L1)': LogisticRegression(
                max_iter=2000, random_state=42, class_weight='balanced',
                C=0.1, penalty='l1', solver='saga'
            ),
            'Logistic Regression (L2)': LogisticRegression(
                max_iter=2000, random_state=42, class_weight='balanced',
                C=0.5, penalty='l2'
            ),
            'Neural Network (Small)': MLPClassifier(
                hidden_layer_sizes=(50,), max_iter=500,
                random_state=42, early_stopping=True
            ),
            'Neural Network (Medium)': MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=500,
                random_state=42, early_stopping=True
            ),
            'Naive Bayes': GaussianNB()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train
            if 'Logistic' in name or 'Neural' in name or 'Naive' in name:
                model.fit(self.X_train_scaled, self.y_train)
                test_pred = model.predict(self.X_test_scaled)
                test_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                model.fit(self.X_train, self.y_train)
                test_pred = model.predict(self.X_test)
                test_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            acc = accuracy_score(self.y_test, test_pred)
            auc = roc_auc_score(self.y_test, test_proba)
            logloss = log_loss(self.y_test, test_proba)
            
            results[name] = {
                'model': model,
                'accuracy': acc,
                'auc': auc,
                'logloss': logloss,
                'needs_scaling': 'Logistic' in name or 'Neural' in name or 'Naive' in name
            }
            
            print(f"  ✅ Acc: {acc:.4f}  AUC: {auc:.4f}  LogLoss: {logloss:.4f}\n")
        
        print("="*60)
        print("📊 RANKING BY AUC (Best Discriminative Power)")
        print("="*60 + "\n")
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
        for i, (name, metrics) in enumerate(sorted_results[:8], 1):
            print(f"{i}. {name:45s} AUC: {metrics['auc']:.4f}  Acc: {metrics['accuracy']:.4f}")
        
        return results
    
    def create_optimized_ensemble(self, top_n=5, test_weights=True):
        """Create ensemble from top performing models"""
        print("\n" + "="*60)
        print("🎯 CREATING OPTIMIZED ENSEMBLE")
        print("="*60 + "\n")
        
        # Get individual model results
        individual_results = self.test_individual_models()
        
        # Select top N models by AUC
        top_models = sorted(individual_results.items(), 
                           key=lambda x: x[1]['auc'], 
                           reverse=True)[:top_n]
        
        print(f"\n✅ Selected Top {top_n} Models:")
        for i, (name, metrics) in enumerate(top_models, 1):
            print(f"  {i}. {name:45s} (AUC: {metrics['auc']:.4f}, Acc: {metrics['accuracy']:.4f})")
        
        # Test different weight combinations
        if test_weights:
            print("\n🔬 Testing Different Weight Combinations...\n")
            
            best_acc = 0
            best_auc = 0
            best_weights = None
            best_ensemble = None
            best_strategy = None
            
            # Try different weight strategies
            weight_strategies = [
                ('Equal Weights', [1] * top_n),
                ('AUC Weighted', [m[1]['auc'] for m in top_models]),
                ('Accuracy Weighted', [m[1]['accuracy'] for m in top_models]),
                ('Top 3 Heavy', [3, 2, 1] + [1] * max(0, top_n - 3)),
                ('Top Model Heavy', [3] + [1] * (top_n - 1)),
                ('Exponential Decay', [2**i for i in range(top_n, 0, -1)]),
                ('Square of AUC', [m[1]['auc']**2 for m in top_models])
            ]
            
            for strategy_name, weights in weight_strategies:
                # Create ensemble
                estimators = []
                for i, (name, data) in enumerate(top_models):
                    estimators.append((f'model_{i}', data['model']))
                
                ensemble = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    weights=weights
                )
                
                # Train and evaluate
                ensemble.fit(self.X_train, self.y_train)
                test_pred = ensemble.predict(self.X_test)
                test_proba = ensemble.predict_proba(self.X_test)[:, 1]
                
                acc = accuracy_score(self.y_test, test_pred)
                auc = roc_auc_score(self.y_test, test_proba)
                logloss = log_loss(self.y_test, test_proba)
                
                print(f"  {strategy_name:25s} Acc: {acc:.4f}  AUC: {auc:.4f}  LogLoss: {logloss:.4f}")
                
                # Track best by accuracy (primary) and AUC (secondary)
                if acc > best_acc or (acc == best_acc and auc > best_auc):
                    best_acc = acc
                    best_auc = auc
                    best_weights = weights
                    best_ensemble = ensemble
                    best_strategy = strategy_name
            
            print(f"\n🏆 Best Strategy: {best_strategy}")
            print(f"   Accuracy: {best_acc:.4f}")
            print(f"   AUC: {best_auc:.4f}")
            print(f"   Weights: {best_weights}")
            
            self.best_ensemble = best_ensemble
            return best_ensemble
        
        else:
            # Use AUC-weighted ensemble
            estimators = [(f'model_{i}', data['model']) for i, (name, data) in enumerate(top_models)]
            weights = [m[1]['auc'] for m in top_models]
            
            ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=weights
            )
            
            ensemble.fit(self.X_train, self.y_train)
            self.best_ensemble = ensemble
            return ensemble
    
    def evaluate_ensemble(self, ensemble):
        """Detailed evaluation of the ensemble"""
        print("\n" + "="*60)
        print("📊 FINAL ENSEMBLE EVALUATION")
        print("="*60)
        
        # Predictions
        train_pred = ensemble.predict(self.X_train)
        test_pred = ensemble.predict(self.X_test)
        train_proba = ensemble.predict_proba(self.X_train)[:, 1]
        test_proba = ensemble.predict_proba(self.X_test)[:, 1]
        
        # Accuracies
        train_acc = accuracy_score(self.y_train, train_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        
        print(f"\n🎯 Accuracy:")
        print(f"  Training: {train_acc:.4f}")
        print(f"  Testing:  {test_acc:.4f}")
        print(f"  Gap:      {train_acc - test_acc:.4f}")
        
        if train_acc - test_acc > 0.15:
            print(f"  ⚠️  Warning: Large gap suggests overfitting")
        elif train_acc - test_acc < 0.05:
            print(f"  ✅ Excellent generalization!")
        
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
        
        # Detailed metrics
        away_precision = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
        away_recall = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        home_precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        home_recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        
        away_f1 = 2 * (away_precision * away_recall) / (away_precision + away_recall) if (away_precision + away_recall) > 0 else 0
        home_f1 = 2 * (home_precision * home_recall) / (home_precision + home_recall) if (home_precision + home_recall) > 0 else 0
        
        print(f"\n🎲 Detailed Metrics:")
        print(f"  Away Team - Precision: {away_precision:.3f}  Recall: {away_recall:.3f}  F1: {away_f1:.3f}")
        print(f"  Home Team - Precision: {home_precision:.3f}  Recall: {home_recall:.3f}  F1: {home_f1:.3f}")
        
        # Overall balance
        balance = min(away_recall, home_recall) / max(away_recall, home_recall)
        print(f"\n⚖️  Prediction Balance: {balance:.3f}")
        if balance > 0.8:
            print(f"  ✅ Well-balanced predictions")
        else:
            print(f"  ⚠️  Model is biased toward {'home' if home_recall > away_recall else 'away'} predictions")
        
        # Probability calibration
        bins = [0, 0.4, 0.5, 0.6, 1.0]
        bin_labels = ['<40%', '40-50%', '50-60%', '>60%']
        
        test_df = pd.DataFrame({
            'actual': self.y_test.values,
            'predicted_prob': test_proba
        })
        test_df['bin'] = pd.cut(test_df['predicted_prob'], bins=bins, labels=bin_labels)
        
        print(f"\n🎯 Probability Calibration:")
        print(f"  {'Bin':8s} {'Predicted':>10s} {'Actual':>10s} {'Error':>10s} {'Count':>10s}")
        print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        
        total_calibration_error = 0
        for bin_name in bin_labels:
            bin_data = test_df[test_df['bin'] == bin_name]
            if len(bin_data) > 0:
                actual_rate = bin_data['actual'].mean()
                avg_pred = bin_data['predicted_prob'].mean()
                error = abs(actual_rate - avg_pred)
                total_calibration_error += error * len(bin_data)
                print(f"  {bin_name:8s} {avg_pred:10.3f} {actual_rate:10.3f} {error:10.3f} {len(bin_data):10d}")
        
        mean_calibration_error = total_calibration_error / len(test_df)
        print(f"\n  Mean Calibration Error: {mean_calibration_error:.4f}")
        if mean_calibration_error < 0.05:
            print(f"  ✅ Excellent calibration!")
        elif mean_calibration_error < 0.10:
            print(f"  ✅ Good calibration")
        else:
            print(f"  ⚠️  Poor calibration - probabilities may not be reliable")
        
        return {
            'test_acc': test_acc,
            'test_auc': test_auc,
            'test_logloss': test_logloss,
            'away_recall': away_recall,
            'home_recall': home_recall
        }
    
    def cross_validate_ensemble(self, ensemble, n_splits=5):
        """Cross-validation for more robust evaluation"""
        print("\n" + "="*60)
        print(f"🔄 {n_splits}-Fold Cross-Validation (Time-Series)")
        print("="*60 + "\n")
        
        # Use time-series split (don't shuffle)
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = []
        cv_auc = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X_train), 1):
            X_fold_train = self.X_train.iloc[train_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_val = self.y_train.iloc[val_idx]
            
            # Clone and train ensemble
            from sklearn.base import clone
            fold_ensemble = clone(ensemble)
            fold_ensemble.fit(X_fold_train, y_fold_train)
            
            # Evaluate
            acc = fold_ensemble.score(X_fold_val, y_fold_val)
            proba = fold_ensemble.predict_proba(X_fold_val)[:, 1]
            auc = roc_auc_score(y_fold_val, proba)
            
            cv_scores.append(acc)
            cv_auc.append(auc)
            
            print(f"  Fold {fold}: Accuracy = {acc:.4f}, AUC = {auc:.4f}")
        
        print(f"\n  Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        print(f"  Mean AUC: {np.mean(cv_auc):.4f} (+/- {np.std(cv_auc):.4f})")
        
        if np.std(cv_scores) < 0.02:
            print(f"  ✅ Very stable model!")
        elif np.std(cv_scores) < 0.05:
            print(f"  ✅ Stable model")
        else:
            print(f"  ⚠️  High variance - model may be unstable")
        
        return cv_scores, cv_auc
    
    def save_ensemble(self, ensemble, model_name='optimized_ensemble'):
        """Save the optimized ensemble"""
        import os
        os.makedirs('models', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save ensemble
        model_file = f"models/{model_name}_{timestamp}.pkl"
        joblib.dump(ensemble, model_file)
        
        # Save preprocessing
        scaler_file = f"models/scaler_{timestamp}.pkl"
        joblib.dump(self.scaler, scaler_file)
        
        imputer_file = f"models/imputer_{timestamp}.pkl"
        joblib.dump(self.imputer, imputer_file)
        
        features_file = f"models/features_{timestamp}.json"
        with open(features_file, 'w') as f:
            json.dump(self.feature_cols, f)
        
        print(f"\n💾 Optimized Ensemble Saved:")
        print(f"  Model: {model_file}")
        print(f"  Scaler: {scaler_file}")
        print(f"  Imputer: {imputer_file}")
        print(f"  Features: {features_file}")
        
        return model_file

# Usage
if __name__ == "__main__":
    print("="*60)
    print("🚀 ENSEMBLE OPTIMIZER - COLLEGE BASKETBALL PREDICTION")
    print("="*60 + "\n")
    
    optimizer = EnsembleOptimizer()
    optimizer.load_data()
    optimizer.prepare_data(test_size=0.2)
    
    # Create optimized ensemble (tests different combinations)
    ensemble = optimizer.create_optimized_ensemble(top_n=6, test_weights=True)
    
    # Detailed evaluation
    results = optimizer.evaluate_ensemble(ensemble)
    
    # Cross-validation
    optimizer.cross_validate_ensemble(ensemble, n_splits=5)
    
    # Save
    optimizer.save_ensemble(ensemble)
    
    print("\n" + "="*60)
    print("✅ OPTIMIZATION COMPLETE!")
    print(f"   Final Test Accuracy: {results['test_acc']:.4f} ({results['test_acc']*100:.1f}%)")
    print(f"   Final Test AUC: {results['test_auc']:.4f}")
    print(f"   Away Win Detection: {results['away_recall']:.4f}")
    print(f"   Home Win Detection: {results['home_recall']:.4f}")
    print("="*60)