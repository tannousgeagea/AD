import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.layers import (Dense, Conv2D, Conv2DTranspose, 
                                     MaxPool2D, BatchNormalization, LeakyReLU, Flatten)



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# set seed
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


def LeNET(input_shape=(128, 128, 1), latent_dim=32):
    inputs = keras.layers.Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=3, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    
    x = Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    
    x = Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPool2D(pool_size=(1, 1))(x)
    
    x = Flatten()(x)
    x = Dense(units=latent_dim, use_bias=False)(x)
    
    model = keras.Model(inputs, x, name='SVDD-NET')
    
    return model


def encoder(input_shape=(128, 128, 1), latent_dim=32):
    inputs = keras.layers.Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=3, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    
    x = Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    
    x = Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPool2D(pool_size=(1, 1))(x)
    out_shape = x.shape
    
    x = Flatten()(x)
    x = Dense(units=latent_dim, use_bias=False)(x)
    
    model = keras.Model(inputs, x, name='encoder')
    
    return model, out_shape



def decoder(input_shape, latent_dim=32):
    inputs = keras.layers.Input(shape=(latent_dim, ))
    x = Dense(units=int(input_shape[1] * input_shape[2] * input_shape[3]), use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    
    x = keras.layers.Reshape(target_shape=input_shape[1:])(x)
    x = Conv2DTranspose(filters=128, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = keras.layers.UpSampling2D(size=(1, 1))(x)
    
    x = Conv2DTranspose(filters=64, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = keras.layers.UpSampling2D(size=(2, 2))(x)
    
    x = Conv2DTranspose(filters=32, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = keras.layers.UpSampling2D(size=(2, 2))(x)
    
    x = Conv2DTranspose(filters=1, kernel_size=3, padding='same', activation='sigmoid', use_bias=False)(x)
    
    model = keras.Model(inputs, x, name='decoder')
    
    return model

def AENet(input_shape=(128, 128, 1), latent_dim=32):

    inputs = keras.layers.Input(shape=input_shape)
    enc, shape = encoder(input_shape, latent_dim)
    dec = decoder(shape, latent_dim)
    
    z = enc(inputs)
    reconstructions = dec(z)
    
    model = keras.Model(inputs, reconstructions, name='AENet')
    
    return model

class AETrainer(keras.models.Model):
    def __init__(self, input_shape, latent_dim, lr=0.001, lr_milestones=[50], **kwargs):
        super(AETrainer, self).__init__(**kwargs)
        
        self.latent_dim = latent_dim
        self.SHAPE = input_shape
        self.lr = lr
        self.lr_milestones = lr_milestones
        
        self.encoder, out_shape = encoder(self.SHAPE, self.latent_dim)
        self.decoder = decoder(out_shape, self.latent_dim)
        
        self.loss_tracker = keras.metrics.Mean(name='loss')
        
    @property
    def metrics(self):
        return [
            self.loss_tracker,
        ]
        
    @tf.function
    def train_step(self, datasets):
        with tf.GradientTape() as tape:
            z = self.encoder(datasets)
            reconstructions = self.decoder(z)
            
            loss = tf.reduce_mean(
                tf.reduce_sum(
                    mse(datasets, reconstructions), axis=(1, 2)
                )
            )
            
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        
        return {
            "loss": self.loss_tracker.result()
        }
    
    
class SVDDTrainer():
    def __init__(self, objective='one-class', R=0, nu=0.1, net=None, c=None,  latent_dim=16, input_shape=(128, 128, 1)):
        
        self.latent_dim = latent_dim
        self.SHAPE = input_shape
        self.net = net
        self.c = c
        self.R = tf.constant(R, dtype='float32')
        self.nu = nu
        self.objective = objective
        
        self.warm_up_n_epochs = 10
        
        if self.net is None:
            self.net = LeNET(input_shape=self.SHAPE, latent_dim=self.latent_dim)
            
        self.loss_tracker = keras.metrics.Mean(name='loss')
        
    @property
    def metrics(self):
        return [
            self.loss_tracker,
        ]
    
    @tf.function
    def train_step(self, datasets):
        
        with tf.GradientTape() as tape:
            outputs = self.net(datasets, training=True)
            dist = tf.reduce_sum(tf.square(outputs - self.c), axis=1)
            
            if self.objective=='soft-boundry':
                score = dist - self.R**2
                penalty = tf.reduce_mean(tf.maximum(score, tf.zeros_like(score)))
                loss = self.R ** 2 + (1 / self.nu) * penalty

            else:
                loss = tf.reduce_mean(dist)
                

        grads = tape.gradient(loss, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        
        return dist, loss
    
    
    def pretrain(self, datasets, ae_epochs, batch_size, ae_lr, ae_lr_milestones, model=None):
        # logger.getLogger()
        if model is None:
            aetrainer = AETrainer(self.SHAPE, self.latent_dim, ae_lr, ae_lr_milestones)
            value = []
            for i in range(len(ae_lr_milestones) + 1):
                value.append(ae_lr)
                ae_lr = ae_lr*0.1

            lr_schedule = schedules.PiecewiseConstantDecay(ae_lr_milestones, value)
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
            aetrainer.compile(optimizer=optimizer)

            # Training
            logger.info('Starting Pretraining...')
            start_time = time.time()
            aetrainer.fit(datasets, epochs=ae_epochs, batch_size=batch_size)

            self.pre_train_time = time.time() - start_time
            logger.info('Pretraining time: %.3f' % self.pre_train_time)
            logger.info('Finished Pretraining.')

            self.net.set_weights(aetrainer.encoder.get_weights())
            
        else:
            logger.info('Loading Weights ....')
            self.net.set_weights(model.get_weights())
            logger.info('Finished Pretraining.')
    
    
    def train(self, datasets, epochs=100, lr=0.001, lr_milestones=[50]):
        
        value = []
        for i in range(len(lr_milestones) + 1):
            value.append(lr)
            lr = lr*0.1
            
        lr_schedule = schedules.PiecewiseConstantDecay(lr_milestones, value)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self._init_c(datasets)
            logger.info('Center c initialized.')
            
        # Training
        logger.info('Starting training...')
        start_time = time.time()
        
        for epoch in range(epochs):
            
            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            
            for x in datasets:
                dist, loss = self.train_step(x)
                
                
                if self.objective == 'soft-boundary' and epoch >= self.warm_up_n_epochs:
                    self.R.data = tf.constant(self._get_R(dist, self.nu))
                
                loss_epoch += loss
                n_batches += 1
                
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, epochs, epoch_train_time, loss_epoch / n_batches))
            
        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')
        
    
    
    def _init_c(self, X, eps=1e-1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        nSamples = 0
        c = np.zeros(self.latent_dim)
        
        for x in X:
            outputs = self.net(x)
            nSamples += outputs.shape[0]
            c += tf.reduce_sum(outputs, axis=0).numpy()
        
        c /= nSamples
        
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
    
    def _get_R(self, dist, nu):
        return np.quantile(np.sqrt(dist), 1 - nu)