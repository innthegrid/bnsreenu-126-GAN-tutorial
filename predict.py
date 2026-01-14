# Generate an image for a specific pont in the latent space

from keras.models import load_model
from numpy import asarray
from matplotlib import pyplot
from numpy.random import randn

# Load model
model = load_model('generator_model.h5')

### Create input vector ###

# To create the same image, supply the same vector each time
# E.g., Vector of all 0s
# vector = asarray([[0. for _ in range(100)]])

# To create a random image each time, generate a vector of random numbers
vector = randn(100)
vector = vector.reshape(1, 100)

#####

# Generate image
X = model.predict(vector)

# Plot the result
pyplot.imshow(X[0, :, :, 0], cmap='gray_r')
pyplot.show()