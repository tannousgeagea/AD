'''
Anomaly Detection using Variational AutoEncoder with tensorflow

'''

# import packages
import numpy as np
import os
from random import shuffle
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.losses import mse, binary_crossentropy

# set seed
SEED = 123
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Create a sampling layer
class Sampling(layers.Layer):
    'using z_mean and z_log_var to sample z'
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]  # batch size
        dim = tf.shape(z_mean)[1]    # latent space dimension
        eps = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(z_log_var * 0.5) * eps

# Build a Convolution layers
def create_conv_layers(configuration, inputs, transpose=False):
    x = inputs
    for config in configuration:
        # config = [[filter, kernel_size, stride], ...]
        
        # for decoder
        if transpose:
            x = layers.Conv2DTranspose(
                filters=config[0], 
                kernel_size=config[1],
                strides=config[2],
                activation='relu',
                padding='same',
                )(x)
            
            
        # for encoder
        else:
            x = layers.Conv2D(
                filters=config[0], 
                kernel_size=config[1],
                strides=config[2],
                activation='relu',
                padding='same',
                )(x)
            
            
        
    return x
 
# Build the Encoder       
def get_encoder(configuration, input_shape, latent_dim):
    inputs = keras.Input(shape=input_shape)
    x = create_conv_layers(configuration, inputs)
    output_shape = x.shape
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    # x = layers.Dense(256, activation='relu')(x)
    # x = layers.Dense(128, activation='relu')(x)
    # x = layers.Dense(64, activation='relu')(x)
    
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    
    return output_shape, keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

# Build the Decoder
def get_decoder(configuration, input_shape, latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))
    shape = input_shape[1] * input_shape[2] * input_shape[3]
    x = layers.Dense(shape, activation='relu')(latent_inputs)
    x = layers.Reshape(input_shape[1:])(x)
    configuration = reversed(configuration)
    x = create_conv_layers(configuration, x, transpose=True)
    output = layers.Conv2D(
        filters=1,
        kernel_size=1,
        strides=1,
        padding='same',
        activation='sigmoid'
        )(x)
    
    return keras.Model(latent_inputs, output, name="decoder")

# Build a Variational AutoEncoder
def get_vae(encoder, decoder, input_shape):
    inputs = keras.Input(shape=input_shape)
    _, _, z = encoder(inputs)
    reconstructions = decoder(z)
    return keras.Model(inputs, reconstructions, name="vae")

# Define VAE as a model with a  training step
class VAETrainer(keras.models.Model):
    def __init__(self, config, 
                 input_shape,
                 latent_dim, **kwargs):
        super(VAETrainer, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        
        self.totol_loss_tracker = keras.metrics.Mean(name='totol_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
        
        self.shape, self.encoder = get_encoder(config, input_shape, self.latent_dim)
        self.decoder = get_decoder(config, self.shape, self.latent_dim)
        self.vae = get_vae(self.encoder, self.decoder, input_shape)
        
    @property
    def metrics(self):
        return [
            self.totol_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function
    def train_step(self, datasets):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(datasets)
            reconstructions = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    mse(datasets, reconstructions), axis=(1, 2)
                )
            )
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = 5 * reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.totol_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return{
            'loss': self.totol_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }
