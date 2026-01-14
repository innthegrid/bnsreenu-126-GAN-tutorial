# Load the generator model and generating images

from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot as plt

# Generate points in latent space as input for the genereator
# latent_dim is the size of the input vector
def generate_latent_points(latent_dim, n_samples):
  # Generate points in the latent space
  x_input = randn(latent_dim * n_samples)

  # Reshape into a batch of inputs for the network
  x_input = x_input.reshape(n_samples, latent_dim)

  return x_input

# Create and save a plot of generated images (reversed grayscale)
def save_plot(examples, n):
  # Plot images
  for i in range(n * n):
    # Define subplot
    plt.subplot(n, n, 1 + i)

    # Turn off axis
    plt.axis('off')

    # Plot raw pixel data
    plt.imshow(examples[i, :, :, 0], cmap='gray_r')
  plt.show()

# Load model
model = load_model('generator_model.h5')

# Generate 16 images, each image provide a vector of size 100 as input
latent_points = generate_latent_points(100, 16)

# Generate images
X = model.predict(latent_points)

# Plot the result on 4x4 grid (16 images)
save_plot(X, 4)