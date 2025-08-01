import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class MLCollatzPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.training_data = None
        self.is_trained = False
        
    def collatz_steps(self, n):
        """Calculate actual number of steps for Collatz sequence"""
        if n <= 0:
            return 0
        
        steps = 0
        while n != 1:
            if n % 2 == 0:
                n = n // 2
            else:
                n = 3 * n + 1
            steps += 1
            
            if steps > 10000:  # Prevent infinite loops
                return steps
                
        return steps
    
    def extract_features(self, numbers):
        """Extract various features from numbers for ML models"""
        if isinstance(numbers, (int, float)):
            numbers = [numbers]
        
        features = []
        
        for n in numbers:
            n = int(n)
            feature_dict = {}
            
            # Basic features
            feature_dict['n'] = n
            feature_dict['log_n'] = np.log(max(n, 1))
            feature_dict['sqrt_n'] = np.sqrt(n)
            feature_dict['bit_length'] = n.bit_length()
            
            # Number theory features
            feature_dict['is_power_of_2'] = (n & (n - 1)) == 0
            feature_dict['trailing_zeros'] = (n & -n).bit_length() - 1 if n > 0 else 0
            feature_dict['popcount'] = bin(n).count('1')  # Number of 1s in binary
            
            # Divisibility features
            feature_dict['div_by_3'] = n % 3 == 0
            feature_dict['mod_3'] = n % 3
            feature_dict['mod_4'] = n % 4
            feature_dict['mod_8'] = n % 8
            feature_dict['mod_16'] = n % 16
            
            # Pattern features
            feature_dict['alternating_bits'] = self._alternating_bit_pattern(n)
            feature_dict['consecutive_ones'] = self._max_consecutive_ones(n)
            feature_dict['consecutive_zeros'] = self._max_consecutive_zeros(n)
            
            # Heuristic features based on Collatz behavior
            feature_dict['odd_divisors'] = self._count_odd_divisors(n)
            feature_dict['largest_odd_factor'] = self._largest_odd_factor(n)
            
            # Statistical features
            binary_str = bin(n)[2:]
            feature_dict['binary_entropy'] = self._binary_entropy(binary_str)
            feature_dict['digit_variance'] = np.var([int(d) for d in str(n)])
            
            features.append(feature_dict)
        
        df = pd.DataFrame(features)
        self.feature_names = df.columns.tolist()
        return df.values
    
    def _alternating_bit_pattern(self, n):
        """Check for alternating bit patterns"""
        binary = bin(n)[2:]
        alternations = sum(1 for i in range(len(binary)-1) 
                          if binary[i] != binary[i+1])
        return alternations / max(len(binary) - 1, 1)
    
    def _max_consecutive_ones(self, n):
        """Find maximum consecutive 1s in binary representation"""
        binary = bin(n)[2:]
        max_ones = 0
        current_ones = 0
        for bit in binary:
            if bit == '1':
                current_ones += 1
                max_ones = max(max_ones, current_ones)
            else:
                current_ones = 0
        return max_ones
    
    def _max_consecutive_zeros(self, n):
        """Find maximum consecutive 0s in binary representation"""
        binary = bin(n)[2:]
        max_zeros = 0
        current_zeros = 0
        for bit in binary:
            if bit == '0':
                current_zeros += 1  
                max_zeros = max(max_zeros, current_zeros)
            else:
                current_zeros = 0
        return max_zeros
    
    def _count_odd_divisors(self, n):
        """Count number of odd divisors"""
        count = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                if i % 2 == 1:
                    count += 1
                if i != n // i and (n // i) % 2 == 1:
                    count += 1
            i += 1
        return count
    
    def _largest_odd_factor(self, n):
        """Find largest odd factor"""
        while n % 2 == 0:
            n = n // 2
        return n
    
    def _binary_entropy(self, binary_str):
        """Calculate entropy of binary representation"""
        if not binary_str:
            return 0
        p1 = binary_str.count('1') / len(binary_str)
        p0 = 1 - p1
        if p1 == 0 or p0 == 0:
            return 0
        return -(p1 * np.log2(p1) + p0 * np.log2(p0))
    
    def generate_training_data(self, max_n=10000, sample_fraction=1.0):
        """Generate training dataset"""
        print(f"Generating training data for numbers 1 to {max_n}")
        
        # Generate numbers to compute
        if sample_fraction < 1.0:
            # Random sampling for large datasets
            n_samples = int(max_n * sample_fraction)
            numbers = np.random.choice(range(1, max_n + 1), n_samples, replace=False)
        else:
            numbers = range(1, max_n + 1)
        
        data = []
        for i, n in enumerate(numbers):
            steps = self.collatz_steps(n)
            data.append({'n': n, 'steps': steps})
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(numbers) if hasattr(numbers, '__len__') else max_n}")
        
        self.training_data = pd.DataFrame(data)
        print(f"Generated {len(self.training_data)} training samples")
        return self.training_data
    
    def prepare_ml_data(self):
        """Prepare features and targets for ML models"""
        if self.training_data is None:
            raise ValueError("No training data available. Run generate_training_data first.")
        
        # Extract features
        print("Extracting features...")
        X = self.extract_features(self.training_data['n'].values)
        y = self.training_data['steps'].values
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Features: {self.feature_names}")
        
        return X, y
    
    def train_models(self, X, y, test_size=0.2, random_state=42):
        """Train multiple ML models"""
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'SVM': SVR(kernel='rbf', C=100, gamma='scale'),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=random_state),
            'Polynomial Ridge': Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=10.0))
            ])
        }
        
        print("Training models...")
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Scale features for models that need it
            if name in ['SVM', 'Neural Network', 'Lasso Regression', 'Ridge Regression']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                self.scalers[name] = scaler
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Train model
            try:
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Calculate metrics
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                # Cross-validation score
                try:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                              cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
                    cv_mae = -cv_scores.mean()
                except:
                    cv_mae = None
                
                results[name] = {
                    'model': model,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'cv_mae': cv_mae,
                    'predictions_test': y_pred_test
                }
                
                print(f"  Test MAE: {test_mae:.2f}, Test R²: {test_r2:.3f}")
                
            except Exception as e:
                print(f"  Failed to train {name}: {str(e)}")
                continue
        
        self.models = {name: result['model'] for name, result in results.items()}
        self.is_trained = True
        
        # Store test data for analysis
        self.X_test = X_test
        self.y_test = y_test
        
        return results, (X_train, X_test, y_train, y_test)
    
    def predict(self, n, model_name='Random Forest'):
        """Make prediction for a single number"""
        if not self.is_trained:
            raise ValueError("Models not trained yet!")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Available: {list(self.models.keys())}")
        
        # Extract features
        X = self.extract_features([n])
        
        # Scale if needed
        if model_name in self.scalers:
            X = self.scalers[model_name].transform(X)
        
        # Make prediction
        prediction = self.models[model_name].predict(X)[0]
        return max(int(round(prediction)), 0)
    
    def predict_ensemble(self, n, models=None):
        """Make ensemble prediction using multiple models"""
        if models is None:
            models = ['Random Forest', 'XGBoost', 'Gradient Boosting']
        
        predictions = []
        for model_name in models:
            if model_name in self.models:
                pred = self.predict(n, model_name)
                predictions.append(pred)
        
        if not predictions:
            raise ValueError("No valid models for ensemble")
        
        # Return median as ensemble prediction
        return int(np.median(predictions))
    
    def analyze_feature_importance(self, model_name='Random Forest'):
        """Analyze feature importance"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return feature_importance
        else:
            print(f"Model {model_name} doesn't support feature importance")
            return None
    
    def plot_results(self, results):
        """Plot comprehensive analysis of results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Model comparison
        ax = axes[0, 0]
        model_names = list(results.keys())
        test_maes = [results[name]['test_mae'] for name in model_names]
        bars = ax.bar(model_names, test_maes)
        ax.set_title('Model Comparison (Test MAE)')
        ax.set_ylabel('Mean Absolute Error')
        ax.tick_params(axis='x', rotation=45)
        
        # Color bars by performance
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # R² scores
        ax = axes[0, 1]
        test_r2s = [results[name]['test_r2'] for name in model_names]
        ax.bar(model_names, test_r2s, color='skyblue')
        ax.set_title('Model R² Scores')
        ax.set_ylabel('R² Score')
        ax.tick_params(axis='x', rotation=45)
        
        # Predictions vs Actual (best model)
        ax = axes[0, 2]
        best_model = min(results.keys(), key=lambda x: results[x]['test_mae'])
        y_pred = results[best_model]['predictions_test']
        ax.scatter(self.y_test, y_pred, alpha=0.5)
        ax.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Steps')
        ax.set_ylabel('Predicted Steps')
        ax.set_title(f'Predictions vs Actual ({best_model})')
        
        # Feature importance
        ax = axes[1, 0]
        feature_importance = self.analyze_feature_importance(best_model)
        if feature_importance is not None:
            top_features = feature_importance.head(10)
            ax.barh(top_features['feature'], top_features['importance'])
            ax.set_title(f'Feature Importance ({best_model})')
            ax.set_xlabel('Importance')
        
        # Error distribution
        ax = axes[1, 1]
        errors = np.abs(self.y_test - y_pred)
        ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.set_yscale('log')
        
        # Learning curves (train vs test error)
        ax = axes[1, 2]
        train_maes = [results[name]['train_mae'] for name in model_names]
        ax.scatter(train_maes, test_maes)
        for i, name in enumerate(model_names):
            ax.annotate(name, (train_maes[i], test_maes[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax.plot([0, max(train_maes)], [0, max(train_maes)], 'r--', alpha=0.5)
        ax.set_xlabel('Train MAE')
        ax.set_ylabel('Test MAE')
        ax.set_title('Overfitting Analysis')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\nModel Performance Summary:")
        print("=" * 80)
        performance_df = pd.DataFrame({
            'Model': model_names,
            'Test_MAE': test_maes,
            'Test_R2': test_r2s,
            'Train_MAE': [results[name]['train_mae'] for name in model_names],
            'CV_MAE': [results[name]['cv_mae'] if results[name]['cv_mae'] else 0 for name in model_names]
        }).sort_values('Test_MAE')
        
        print(performance_df.round(3))

# Example usage and demonstration
if __name__ == "__main__":
    # Create predictor
    predictor = MLCollatzPredictor()
    
    # Generate training data
    predictor.generate_training_data(max_n=8000)  # Smaller dataset for demo
    
    # Prepare ML data
    X, y = predictor.prepare_ml_data()
    
    # Train models
    results, data_splits = predictor.train_models(X, y)
    
    # Test predictions
    test_numbers = [12345, 98765, 123456, 555555]
    print(f"\nPredictions for test numbers:")
    print("-" * 60)
    
    for n in test_numbers:
        print(f"\nNumber: {n}")
        
        # Individual model predictions
        for model_name in ['Random Forest', 'XGBoost', 'Neural Network']:
            if model_name in predictor.models:
                try:
                    pred = predictor.predict(n, model_name)
                    print(f"{model_name:15}: {pred:4d} steps")
                except:
                    print(f"{model_name:15}: Failed")
        
        # Ensemble prediction
        try:
            ensemble_pred = predictor.predict_ensemble(n)
            print(f"{'Ensemble':15}: {ensemble_pred:4d} steps")
        except:
            print(f"{'Ensemble':15}: Failed")
        
        # Actual value for comparison (if computable)
        if n < 100000:
            actual = predictor.collatz_steps(n)
            print(f"{'Actual':15}: {actual:4d} steps")
    
    # Show feature importance
    print(f"\nTop 10 Most Important Features:")
    feature_importance = predictor.analyze_feature_importance()
    if feature_importance is not None:
        print(feature_importance.head(10))
    
    # Plot comprehensive analysis
    predictor.plot_results(results)