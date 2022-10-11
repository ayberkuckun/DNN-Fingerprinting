import numpy as np
from tensorflow import keras

optimizer1 = keras.optimizers.Adam(learning_rate=0.0005)

# Weights for Ensemble Loss
alpha = 1
beta = 1
gamma = 1

# Sensitivity
epsilon = [0.15]
# epsilon = np.arange(0.01, 0.161, 0.015)
# epsilon = np.arange(0.115, 0.161, 0.015)

# Decision threshold
p = 0.75

# Conferrability threshold
threshold = 0.85
