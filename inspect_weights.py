import matplotlib.pyplot as plt
import numpy as np

rbm_features = np.load('features/out.npy')
sampled_features = np.load('features/sampled.npy')

print(f'Euclidean distance: {np.linalg.norm(rbm_features[0] - sampled_features[0])}')

# Creating a pyplot figure
fig = plt.figure(figsize=(1,2))

# Defines the subplot
plt.subplot(1, 2, 1)
plt.imshow(np.reshape(rbm_features[0], (16, 16)) * 127.5 + 127.5, cmap='gray')

# Disabling the axis
plt.axis('off')

# Defines the subplot
plt.subplot(1, 2, 2)
plt.imshow(np.reshape(sampled_features[0], (16, 16)) * 127.5 + 127.5, cmap='gray')

# Disabling the axis
plt.axis('off')

# Showing the plot
plt.show()
