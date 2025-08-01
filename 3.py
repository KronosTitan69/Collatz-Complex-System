import numpy as np

def collatz_steps(n):
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3*n + 1
        steps += 1
    return steps

N = 10000
X = np.arange(2, N+2)
y = np.array([collatz_steps(n) for n in X])

import pandas as pd

df = pd.DataFrame({'n': X, 'steps': y})
df['log_n'] = np.log(df['n'])
df['mod2'] = df['n'] % 2
df['mod3'] = df['n'] % 3
df['step_gradient'] = np.gradient(df['steps'], df['n'])

features = df[['log_n', 'mod2', 'mod3', 'step_gradient']].values
target = df['steps'].values

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.1, random_state=42)

# Define the model
def build_model(input_shape):
    inputs = keras.Input(shape=(input_shape,))
    x = keras.layers.Dense(128, activation='relu')(inputs)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    return model

model = build_model(X_train.shape[1])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.1)

import numpy as np

def mc_predict(model, x, T=100):  # T = number of MC samples
    preds = np.array([model(x, training=True).numpy().flatten() for _ in range(T)])
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std

# Predict on test set
mean_pred, std_pred = mc_predict(model, X_test)

# Probability of exact step count (discretize normal PDF)
def prob_exact(y_true, mean, std):
    # Assume normal distribution at each prediction
    prob = (
        1/(std*np.sqrt(2*np.pi)) *
        np.exp(-0.5 * ((y_true-mean)/std)**2)
    )
    return prob

probs = prob_exact(y_test, mean_pred, std_pred)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.scatter(y_test, mean_pred, alpha=0.3, label='Predicted vs True')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect prediction')
plt.xlabel('True Steps')
plt.ylabel('Predicted Steps')
plt.legend()
plt.title('Collatz Steps: Neural Network Prediction')
plt.show()

# Uncertainty
plt.figure(figsize=(10,6))
plt.errorbar(y_test[:200], mean_pred[:200], yerr=std_pred[:200], fmt='o', alpha=0.5)
plt.xlabel('True Steps')
plt.ylabel('Predicted Steps')
plt.title('Prediction with Uncertainty (MC Dropout)')
plt.show()

# Probability histogram
plt.figure(figsize=(10,6))
plt.hist(probs, bins=30)
plt.xlabel('Probability of exact prediction')
plt.ylabel('Frequency')
plt.title('Distribution of Exact Prediction Probabilities')
plt.show()