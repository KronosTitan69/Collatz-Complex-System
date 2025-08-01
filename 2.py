import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import gamma
import warnings
warnings.filterwarnings('ignore')

class CollatzPredictor:
    def __init__(self, max_n=10000):
        self.max_n = max_n
        self.data = None
        self.models = {}
        self.scalers = {}

    def collatz_steps(self, n):
        """Calculate the number of steps for n to reach 1 in Collatz sequence"""
        if n <= 0:
            return 0
        steps = 0
        original_n = n
        while n != 1:
            if n % 2 == 0:
                n = n // 2
            else:
                n = 3 * n + 1
            steps += 1
            # Safety check to avoid infinite loops
            if steps > 10000:
                return None
        return steps

    def extract_features(self, n):
        """Extract comprehensive features including PDE-inspired ones"""
        if n <= 0:
            return np.zeros(20) # Ensure exactly 20 features are returned for n<=0

        features = []

        # Basic features (4)
        features.append(n)
        features.append(np.log(n + 1))
        features.append(np.log2(n + 1))
        features.append(np.sqrt(n))

        # Binary representation features (4)
        binary = bin(n)[2:]
        features.append(len(binary))  # bit length
        features.append(binary.count('1'))  # number of 1s
        features.append(binary.count('0'))  # number of 0s
        features.append(binary.count('1') / len(binary) if len(binary) > 0 else 0)  # density of 1s

        # Modular arithmetic features (4)
        features.append(n % 3)
        features.append(n % 4)
        features.append(n % 8)
        features.append(n % 16)

        # PDE-inspired features based on the continuous extension
        # These are motivated by the partial differential equation formulation
        # of the Collatz problem in continuous domain

        # Harmonic analysis inspired features (2)
        features.append(np.sin(2 * np.pi * tf.math.log(tf.cast(n + 1, dtype=tf.float32))))
        features.append(np.cos(2 * np.pi * tf.math.log(tf.cast(n + 1, dtype=tf.float32))))

        # Features inspired by the p-adic analysis of Collatz (1)
        # Related to the 2-adic and 3-adic properties
        v2 = 0  # 2-adic valuation
        temp_n = n
        while temp_n % 2 == 0 and temp_n > 0:
            v2 += 1
            temp_n //= 2
        features.append(v2)

        # Features related to the "height" function in dynamical systems (2)
        max_value = n
        temp_n = n
        steps_to_max = 0
        current_steps = 0

        # Calculate trajectory statistics (limited steps for efficiency)
        # Increased loop iterations for better approximation, but keep a limit
        for _ in range(min(1000, n.bit_length() * 20)): # Increased limit
            if temp_n == 1:
                break
            if temp_n % 2 == 0:
                temp_n = temp_n // 2
            else:
                temp_n = 3 * temp_n + 1
            current_steps += 1
            # Add a check to prevent infinite loops
            if current_steps > 100000: # Safety break
                max_value = np.nan # Indicate failure
                steps_to_max = np.nan # Indicate failure
                break
            if temp_n > max_value:
                max_value = temp_n
                steps_to_max = current_steps

        features.append(np.log(max_value + 1) / np.log(n + 1) if n > 1 and not np.isnan(max_value) else (1.0 if n == 1 else 0.0))  # height ratio
        features.append(steps_to_max if not np.isnan(steps_to_max) else 0) # Handle NaN case

        # Fractal dimension inspired feature (1)
        features.append(np.log(n + 1) / np.log(np.log(n + 2)) if n > 1 else 0.0) # Handle small n

        # Features based on the residue classes modulo powers of 2 (1)
        features.append((n % 32) / 32.0)  # normalized residue mod 32

        # Ensure exactly 20 features are returned
        if len(features) != 20:
            # This is a safeguard. If the counts above are correct, this won't be needed.
            #print(f"Warning: Incorrect number of features ({len(features)}) generated for n={n}. Expected 20.")
            # Pad or truncate features to ensure 20
            if len(features) < 20:
                features.extend([0.0] * (20 - len(features)))
            elif len(features) > 20:
                features = features[:20]


        return np.array(features)

    def generate_dataset(self):
        """Generate training dataset with features and target values"""
        print("Generating Collatz dataset...")
        data = []

        for n in range(1, self.max_n + 1):
            if n % 1000 == 0:
                print(f"Processing {n}/{self.max_n}")

            steps = self.collatz_steps(n)
            if steps is not None:
                features = self.extract_features(n)
                # Ensure features is a numpy array before concatenating
                if not isinstance(features, np.ndarray):
                     features = np.array(features)
                data.append(np.concatenate([[n, steps], features]))


        # The columns list should match the concatenated data: ['n', 'steps'] + 20 features
        columns = ['n', 'steps'] + [f'feature_{i}' for i in range(20)]
        self.data = pd.DataFrame(data, columns=columns)
        return self.data

    def create_probabilistic_neural_network(self, input_dim):
        """Create a neural network that outputs mean and variance for probabilistic predictions"""
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2), # Added dropout layer
            layers.Dense(32, activation='relu'),
            # Two outputs: mean and log(variance)
            layers.Dense(2)
        ])
        return model

    def gaussian_nll_loss(self, y_true, y_pred):
        """Negative log-likelihood loss for Gaussian distribution"""
        mean = y_pred[:, 0] # Removed the extra dimension [:, 0:1] -> [:, 0]
        log_var = y_pred[:, 1] # Removed the extra dimension [:, 1:2] -> [:, 1]
        var = tf.exp(log_var)

        # Ensure y_true has the same shape as mean for element-wise operations
        y_true = tf.cast(y_true, dtype=tf.float32) # Cast to float32
        y_true = tf.reshape(y_true, [-1]) # Reshape to match mean

        # Compute negative log-likelihood
        nll = 0.5 * tf.math.log(2. * np.pi) + 0.5 * log_var + 0.5 * tf.square(y_true - mean) / var
        return tf.reduce_mean(nll)

    def train_models(self):
        """Train multiple models for ensemble prediction"""
        if self.data is None:
            self.generate_dataset()

        # Prepare features and target
        feature_cols = [col for col in self.data.columns if col.startswith('feature_')]
        X = self.data[feature_cols].values
        y = self.data['steps'].values

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler

        print("Training models...")

        # 1. Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)
        self.models['random_forest'] = rf_model

        # 2. Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        self.models['gradient_boosting'] = gb_model

        # 3. Neural Network (standard)
        nn_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            alpha=0.01,
            max_iter=1000,
            random_state=42
        )
        nn_model.fit(X_train_scaled, y_train)
        self.models['neural_network'] = nn_model

        # 4. Probabilistic Neural Network
        prob_nn = self.create_probabilistic_neural_network(X_train_scaled.shape[1])
        prob_nn.compile(
            optimizer='adam',
            loss=self.gaussian_nll_loss,
            # Removed 'mae' metric to avoid shape conflicts
        )

        # Train probabilistic model
        prob_nn.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=100,
            batch_size=32,
            verbose=0
        )
        self.models['probabilistic_nn'] = prob_nn

        # Evaluate models
        print("\nModel Performance:")
        for name, model in self.models.items():
            if name == 'probabilistic_nn':
                # For probabilistic model, use mean prediction
                pred = model.predict(X_test_scaled, verbose=0)
                y_pred = pred[:, 0]  # mean predictions
            else:
                y_pred = model.predict(X_test_scaled)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"{name}: MAE = {mae:.3f}, RMSE = {rmse:.3f}")

        return X_test, y_test

    def predict_with_uncertainty(self, n, n_samples=1000):
        """Predict steps with uncertainty quantification"""
        features = self.extract_features(n).reshape(1, -1)
        features_scaled = self.scalers['main'].transform(features)

        predictions = {}

        # Get predictions from all models
        for name, model in self.models.items():
            if name == 'probabilistic_nn':
                # Get mean and variance from probabilistic model
                pred = model.predict(features_scaled, verbose=0)
                mean_pred = pred[0, 0]
                log_var_pred = pred[0, 1]
                var_pred = np.exp(log_var_pred)
                std_pred = np.sqrt(var_pred)

                predictions[name] = {
                    'mean': mean_pred,
                    'std': std_pred,
                    'samples': np.random.normal(mean_pred, std_pred, n_samples)
                }
            else:
                # For other models, use bootstrap sampling for uncertainty
                pred = model.predict(features_scaled)[0]

                # Estimate uncertainty using feature perturbation
                perturbed_preds = []
                for _ in range(100):
                    noise = np.random.normal(0, 0.01, features_scaled.shape)
                    perturbed_features = features_scaled + noise
                    perturbed_pred = model.predict(perturbed_features)[0]
                    perturbed_preds.append(perturbed_pred)

                std_pred = np.std(perturbed_preds)
                predictions[name] = {
                    'mean': pred,
                    'std': std_pred,
                    'samples': np.random.normal(pred, std_pred, n_samples)
                }

        return predictions

    def ensemble_predict(self, n, n_samples=1000):
        """Ensemble prediction combining all models"""
        individual_preds = self.predict_with_uncertainty(n, n_samples)

        # Combine predictions using weighted average
        weights = {
            'random_forest': 0.3,
            'gradient_boosting': 0.3,
            'neural_network': 0.2,
            'probabilistic_nn': 0.2
        }

        ensemble_mean = sum(weights[name] * pred['mean']
                          for name, pred in individual_preds.items())

        # Combine uncertainties
        ensemble_var = sum(weights[name]**2 * pred['std']**2
                          for name, pred in individual_preds.items())
        ensemble_std = np.sqrt(ensemble_var)

        # Generate ensemble samples
        ensemble_samples = sum(weights[name] * pred['samples']
                             for name, pred in individual_preds.items())

        return {
            'ensemble': {
                'mean': ensemble_mean,
                'std': ensemble_std,
                'samples': ensemble_samples
            },
            'individual': individual_preds
        }

    def calculate_step_probabilities(self, n, max_steps=None):
        """Calculate probability distribution over possible step counts"""
        predictions = self.ensemble_predict(n)
        samples = predictions['ensemble']['samples']

        if max_steps is None:
            max_steps = int(np.max(samples)) + 10 if len(samples) > 0 else 10

        # Create probability distribution
        step_counts = np.arange(1, max_steps + 1)
        probabilities = np.zeros(len(step_counts))

        for i, steps in enumerate(step_counts):
            # Use kernel density estimation
            # Using a fixed bandwidth or a more robust method like Scott's Rule or Silverman's Rule could improve this
            # For simplicity, using a fixed bandwidth here
            bandwidth = 5 # This bandwidth might need tuning
            prob = np.sum(np.exp(-0.5 * ((samples - steps) / bandwidth)**2)) / len(samples)
            probabilities[i] = prob

        # Normalize probabilities
        if np.sum(probabilities) > 0:
            probabilities = probabilities / np.sum(probabilities)
        else:
            # Handle case where all probabilities are zero
            probabilities = np.ones(len(step_counts)) / len(step_counts)


        return step_counts, probabilities

    def visualize_predictions(self, test_numbers=None):
        """Visualize model predictions and uncertainties"""
        if test_numbers is None:
            test_numbers = [27, 63, 127, 255, 511, 1023]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, n in enumerate(test_numbers):
            if i >= len(axes):
                break

            # Get actual steps
            actual_steps = self.collatz_steps(n)

            # Get probability distribution
            try:
                steps, probs = self.calculate_step_probabilities(n, max_steps=100) # Limit max_steps for plotting
            except Exception as e:
                 print(f"Could not calculate probabilities for n={n}: {e}")
                 continue


            # Plot probability distribution
            axes[i].bar(steps, probs, alpha=0.7, width=0.8)
            if actual_steps is not None:
                axes[i].axvline(actual_steps, color='red', linestyle='--',
                              linewidth=2, label=f'Actual: {actual_steps}')

            # Get ensemble prediction
            try:
                ensemble_pred = self.ensemble_predict(n)
                mean_pred = ensemble_pred['ensemble']['mean']
                std_pred = ensemble_pred['ensemble']['std']

                axes[i].axvline(mean_pred, color='blue', linestyle='-',
                              linewidth=2, label=f'Predicted: {mean_pred:.1f}±{std_pred:.1f}')
            except Exception as e:
                 print(f"Could not get ensemble prediction for n={n}: {e}")


            axes[i].set_title(f'n = {n}')
            axes[i].set_xlabel('Steps to reach 1')
            axes[i].set_ylabel('Probability')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig

def main():
    """Main function to demonstrate the Collatz predictor"""
    # Initialize predictor
    predictor = CollatzPredictor(max_n=5000)

    # Generate dataset and train models
    print("Starting Collatz trajectory prediction model...")
    predictor.generate_dataset()
    predictor.train_models()

    # Test predictions on specific numbers
    test_numbers = [27, 63, 127, 255, 511, 1023, 2047]
    print(f"\nTesting predictions on: {test_numbers}")

    results = []
    for n in test_numbers:
        actual = predictor.collatz_steps(n)
        try:
            predictions = predictor.ensemble_predict(n)
            ensemble = predictions['ensemble']

            result = {
                'n': n,
                'actual_steps': actual,
                'predicted_mean': ensemble['mean'],
                'predicted_std': ensemble['std'],
                'error': abs(actual - ensemble['mean']) if actual is not None else None
            }
            results.append(result)

            print(f"n={n}: Actual={actual}, Predicted={ensemble['mean']:.1f}±{ensemble['std']:.1f}, Error={result['error']:.1f}" if actual is not None else f"n={n}: Actual={actual}, Predicted={ensemble['mean']:.1f}±{ensemble['std']:.1f}")
        except Exception as e:
            print(f"Prediction failed for n={n}: {e}")


    # Calculate probability distributions
    print(f"\nProbability distributions for exact step counts:")
    for n in [27, 63, 127]:
        try:
            steps, probs = predictor.calculate_step_probabilities(n, max_steps=200)
            actual_steps = predictor.collatz_steps(n)

            # Find probability of actual step count
            if actual_steps is not None and actual_steps > 0 and actual_steps <= len(steps):
                actual_prob = probs[actual_steps - 1]
                print(f"n={n}: P(steps={actual_steps}) = {actual_prob:.4f}")

                # Find most likely step count
                most_likely_steps = steps[np.argmax(probs)]
                max_prob = np.max(probs)
                print(f"n={n}: Most likely steps = {most_likely_steps} (P = {max_prob:.4f})")
            elif actual_steps is not None:
                 print(f"n={n}: Actual steps {actual_steps} is outside the calculated range or is zero.")
            else:
                 print(f"n={n}: Could not calculate actual steps.")

            print()
        except Exception as e:
            print(f"Could not calculate probability distribution for n={n}: {e}")


    # Visualize results
    predictor.visualize_predictions(test_numbers[:6])

    return predictor, results

if __name__ == "__main__":
    predictor, results = main()