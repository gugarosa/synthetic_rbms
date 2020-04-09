import matplotlib.pyplot as plt
import numpy as np

rbm_weights = np.load('weights/vanilla_rbm_0.npy')
sampled_weights = np.load('weights/vanilla_gan.npy')

print(f'Euclidean distance: {np.linalg.norm(rbm_weights[0] - sampled_weights[0])}')

# Creating a pyplot figure
fig = plt.figure(figsize=(1,2))

# Defines the subplot
plt.subplot(1, 2, 1)
plt.imshow(np.reshape(rbm_weights[0], (16, 16)) * 127.5 + 127.5, cmap='gray')

# Disabling the axis
plt.axis('off')

# Defines the subplot
plt.subplot(1, 2, 2)
plt.imshow(np.reshape(sampled_weights[0], (16, 16)) * 127.5 + 127.5, cmap='gray')

# Disabling the axis
plt.axis('off')

# Showing the plot
plt.show()
