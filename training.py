# Keras is a library that acts like a wrapper around TensorFlow, provides simple commands to use TensorFlow

# keras.datasets is a library of pre-packaged sample data
# Import MNIST database, collection of handwritten digits - the "real" data
from keras.datasets import mnist

# A layer is a specific mathematical transformation that data passes through
# Input - a placeholder for the starting data
# Dense - a standard brick; every neuron in this layer connects to every neuron in the next
# Reshape - a morphing brick; changes the data from a flat line into a 2D square
# Dropout - a filter brick; randomly turns off neurons to prevent the model from overfitting
from keras.layers import Input, Dense, Reshape, Flatten

# Standardies the outputs from the previous layer to always have a mean of 0 and a standard deviation of 1
from keras.layers import BatchNormalization

# Standard ReLu turns all negative numbers into zero
# Leaky ReLu allows a small percentage of negative signal to pass through, keeps the GAN "alive"
# Original line - from keras.layers.advanced_activations import LeakyReLU
from keras.layers import LeakyReLU

# Defines the "container" for the layers
# Sequential - A simple, linear stack what data flows through each layer
# Model - More flexible paths
from keras.models import Sequential, Model

# The optimizer is the algorithm that updates the model's weights to make it better; Adam is a popular choice
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np

#####

# Each image in the MNIST dataset is 28x28 pixels.
# Define input image dimensions
img_rows = 28
img_cols = 28

# The images are in grayscale, so each pixel has 1 value describing how bright it is (black to white)
channels = 1

# Creates a tuple defining our image shape
img_shape = (img_rows, img_cols, channels)

#####

# Given an input of noise (latent) vector, the Generator will generate an image
def build_generator():
	# Define the noise (seed) to be a vector (1D) of size 100
	noise_shape = (100,)
  
	# Define the generator network
  # Note: Current GAN is simple - only uses Dense layers (easy to understand, fast to train) - but don't see shapes well
	#       There are more powerful versions (e.g., VGG for super-resolution GAN)
  
	# Creates an empty container to add layers into.
  # Data flows through each layer
	model = Sequential()

	# Each Dense layer adds more parameters (256 -> 512 -> 1024), allowing the model to learn more intricate details.
  
	# Input player (noise_shape 100) and hidden layer (dense(256))
	# Adds a layer into the model
	# Creates a Dense layer with 256 neurons, and expect an input of 100 numbers (noise_shape)
	model.add(Dense(256, input_shape=noise_shape))

	# Alpha is a hyperparameter that controls how much the function passes through negative network inputs
	# If alpha is too small, the signal is too weak and the neurons can still die
	# If alpha is too big, the model becomes too linear and loses its inability to learn complex shapes
	model.add(LeakyReLU(alpha=0.2))

	# Batch normalization keeps a running average of mean and variance of the data
	# Momentum decides how much weight to give to the past (80%) vs. the current data (20%)
	# If momentum is too small, the normalization would jump around wildly with each new batch of data
	# Acts as stabilizer, which allows a higher learning rate
	model.add(BatchNormalization(momentum=0.8))

	# Hidden layer
	model.add(Dense(512))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))

	# Hidden layer
	model.add(Dense(1024))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))

	# np.prod multiples 28x28x1 (from img_shape) = 784
	# The final Dense layer creates 784 outputs - one for every pixel in the MNIST image
	# Tanh activation normalizes all output numbers to be between -1 and 1
	model.add(Dense(np.prod(img_shape), activation='tanh'))

	# Before, the data is a long list of 784 numbers
	# Reshape folds it into a 28x28 grid - turns it into an image
	model.add(Reshape(img_shape))

	# Prints a table showing the layers and number of parameters
	model.summary()

	# Creates the Tensorspace to accept a list of 100 numbers, the size of the seed
	noise = Input(shape=noise_shape)

	# Feeds the noise through the finished model - returns the generated image
	img = model(noise)

	# Wrap the input, layers, and output into a single, usable object called a Model
	return Model(noise, img)

#####

# Given an input image, the Discriminator returns the likelihood of the image being real (binary classification)
def build_discriminator():
	model = Sequential()

	# Disciminator is the opposite of the generator, condenses from 512 -> 256
	# Taking 784 pieces of information and finding the most important features

	# Input layer
	# Dense layers cannot process 2D grids, only understand long lists of numbers
	# Flatten unrolls the 28x28 grid into a single flat line of 784 pixels
	model.add(Flatten(input_shape=img_shape))

	# Hidden layer
	model.add(Dense(512))
	model.add(LeakyReLU(alpha=0.2))

	# No Batch Normalization

	# Hidden layer
	model.add(Dense(256))
	model.add(LeakyReLU(alpha=0.2))

	# Output layer
	# Sigmoid activation normalizes the output between 0 and 1
	# 0 = fake, 1 = real
	model.add(Dense(1, activation='sigmoid'))

	model.summary()

	img = Input(shape=img_shape)
	validity = model(img)

	return Model(img, validity)

#####

# Trains both the discriminator and the generator
# They train one at a time (the other is locked). They take turns in phases in each epoch.
def train(epochs, batch_size=128, save_interval=50):
	# Load the dataset
	# mnist.load_data() returns 4 things: X_train (60k images), y_train (labels), X_test (10k images), y_test (labels)
	# We only need X_train
	(X_train, _), (_, _) = mnist.load_data()
	
	# Convert to float and rescale -1 to 1
	# Standard digital images uses 8-bit integers, so every pixel has value 0 (black) to 255 (white)
	# We subtract and divide by it to linearly scale it to a new range
	# We are using the range -1 to 1 to match tanh activation range
	X_train = (X_train.astype(np.float32) - 127.5) / 127.5
	
	# The input to our generator and discriminator has a shape of 28x28x1
	# Add channels dimension (1)
	X_train = np.expand_dims(X_train, axis=3)
	
	# Default batch_size is 128. Half of this should be real (from MNIST) and half should be fake (from generator)
	half_batch = int(batch_size / 2)

	# Loop through a number of epochs
	for epoch in range(epochs):

		### TRAIN DISCRIMINATOR ###
		
		# Train the discriminator to distinguish between real (1) and fake (0)
		# Select a random batch of real images (MNIST), generate a set of fake images (Generator),
		# feed both into the Discriminator, and finally set the loss for real, fake, and combined

		# Select a random half batch of real images
		# Create a list of half_batch number of random index numbers from 0 to 60k (X_train.shape[0])
		idx = np.random.randint(0, X_train.shape[0], half_batch)
		# Selects the 64 images at the random indices we generates
		imgs = X_train[idx]

		# Create the seeds (input for the generator), based on a normal distribution (0 mean, 1 std)
		# Create half_batch number of vectors, each of size 100
		noise = np.random.normal(0, 1, (half_batch, 100))

		# The generator takes the noise and generates half_batch number of fake images
		# gen_imgs is a NumPy array with shape (half_batch, 28, 28, 1)
		# Because of the tanh activation, every pixel in these 64 images is between -1 and 1
		gen_imgs = generator.predict(noise)

		# Train the discriminator on real and fake images, separately (more effective)
		# Train on real images - input half_batch number of MNIST images
		# Since they are all real, the answer key is an array of 1s (np.ones((half_batch, 1)))
		d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))

		# Train on generated images, with answer key of all 0s
		d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

		# Since we trained the model twice, we average them to get a overall loss
		d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
		
		### TRAIN GENERATOR ###
		
		# Trains the generator to trick the discriminator by labeling fake images as real (1)
		# Discriminator is frozen so that it cannot become more forgiving to reach target, generator has to get better

		# Create batch_size number of noise veectors, based on normal distribution, as input for the generator
		# Output is size (batch size, 100)
		noise = np.random.normal(0, 1, (batch_size, 100))

		# The generator wants the discriminator to label the generated samples as real (ones)
		# We create an array of all ones as the answer key
		# When the discriminator (correctly) labels the fake image as 0, the loss will be huge
		# Since the discriminator is locked, the generator will need to improve
		valid_y = np.array([1] * batch_size)

		# The generator is trained with a combined model (linked with the discriminator)
		# because on its own, the generator doesn't have a loss function to classify real/fake
		g_loss = combined.train_on_batch(noise, valid_y)

		# Print the progress
		print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

		# If at save interal, save the generated image samples
		if epoch % save_interval == 0:
			save_imgs(epoch)

#####

# Save our images for us to view
def save_imgs(epoch):
	# Saving a 5x5 grid - create 25 samples
	r, c = 5, 5
  
	noise = np.random.normal(0, 1, (r * c, 100))
	gen_imgs = generator.predict(noise)

	# Rescale images to range between 0 and 1
	# Before, we used the range of -1 and 1, but viewing libraries expect the range 0-1
	gen_imgs = 0.5 * gen_imgs + 0.5

	fig, axs = plt.subplots(r, c)
	cnt = 0
	for i in range(r):
		for j in range(c):
			axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
			axs[i,j].axis('off')
			cnt += 1
	fig.savefig("images/mnist_%d.png" % epoch)
	plt.close()
	
#####

# We define the optimizer - updates the weights of the network when training
# Hyperparameters (learning rate, momentum)
optimizer = Adam(0.0002, 0.5)

# Build and compile the discriminator first
# Discriminator is an instance of the Keras Model class
discriminator = build_discriminator()
# Compile locks the settings for this model
# We use binary_crossentropy because the discriminator is a binary classifier
# Use the optimizer we defined earlier
# Keeping track of accuracy as a metric for understanding
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Build and compile our generator
generator = build_generator()
# We are only generating images, so we don't need to track any metrics
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

# Defines the input parameter for the model, vector of size 100
z = Input(shape=(100,))
# The fake, generated image
img = generator(z)

# We lock (set to false) the discriminator trainable settings; does not impact the previously compiled settings
# This ensures that when we combine our models, we only train the generator
# Later when we call train, it still trains discriminator with the previous compilation settings
# And trains combined with this setting
discriminator.trainable = False

# Feed the generated image into the discriminator and store its output in valid (real or not)
valid = discriminator(img)

# Combine the models (stack the generator and discriminator)
# Takes the noise as input (z) -> generates images -> determines validity (valid)
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

train(epochs=10000, batch_size=32, save_interval=500)

generator.save('generator_model.h5')