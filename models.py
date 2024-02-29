import tensorflow as tf
from tensorflow.keras import layers

class ProGAN:
    def __init__(self,
                 input_shape,
                 latent_shape,
                 dicrim_shape,
                 ):
        super(ProGAN, self).__init__()
        self.input_shape  = input_shape
        self.latent_shape = latent_shape
        self.dicrim_shape = dicrim_shape
        
    def _make_generator(self, gen_filter):
        return tf.keras.Sequential([
            layers.InputLayer(input_shape=self.latent_shape),
            layers.Dense(units=7*8*7*gen_filter, activation=tf.nn.relu),
            layers.Reshape(target_shape=(7, 8, 7, gen_filter)),
            layers.Conv3DTranspose(
                filters=gen_filter*2, kernel_size=3, strides=(3,3,3), padding='same',
                activation='relu'),
            layers.Conv3DTranspose(
                filters=gen_filter, kernel_size=3, strides=(2,2,2), padding='same',
                activation='relu'),
            layers.Conv3DTranspose(
                filters=1, kernel_size=3, strides=(2,1,1), padding='same'),
        ])
        
    def _make_encoder(self, input_shape, enc_filter, enc_dropout, latent_dim, pool_size=2):
        return tf.keras.Sequential([
            layers.InputLayer(input_shape=input_shape),
            layers.Conv3D(enc_filter, 5, strides=1, padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(enc_dropout),
            layers.Conv3D(enc_filter, 5, strides=1, padding='same'),
            layers.LeakyReLU(),
            layers.MaxPooling3D(pool_size=pool_size),
            layers.Conv3D(int(enc_filter*1.5), 5, strides=1, padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(enc_dropout),
            layers.Conv3D(int(enc_filter*1.5), 5, strides=1, padding='same'),
            layers.LeakyReLU(),
            layers.MaxPooling3D(pool_size=pool_size),
            layers.Conv3D(int(enc_filter*2), 3, strides=1, padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(enc_dropout),
            layers.Conv3D(int(enc_filter*2), 3, strides=1, padding='same'),
            layers.LeakyReLU(),
            layers.MaxPooling3D(pool_size=pool_size),
            layers.Flatten(),
            layers.Dense(latent_dim+latent_dim)
        ])
    
    def _make_decoder(self, dec_filter):
        return tf.keras.Sequential([
            layers.InputLayer(input_shape=self.latent_shape),
            layers.Dense(units=7*8*7*dec_filter, activation=tf.nn.relu),
            layers.Reshape(target_shape=(7, 8, 7, dec_filter)),
            layers.Conv3DTranspose(
                filters=dec_filter*2, kernel_size=3, strides=3, padding='same',
                activation='relu'),
            layers.Conv3DTranspose(
                filters=dec_filter, kernel_size=5, strides=3, padding='same',
                activation='relu'),
            layers.Conv3DTranspose(
                filters=1, kernel_size=5, strides=3, padding='same'),
        ])
    
    def _make_dicriminator(self, dic_filter, dic_dropout):
        return tf.keras.Sequential([
            layers.InputLayer(input_shape=self.dicrim_shape),
            layers.Conv3D(dic_filter, 5, strides=2, padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(dic_dropout),
            layers.Conv3D(int(dic_filter*1.5), 5, strides=2, padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(dic_dropout),
            layers.Conv3D(int(dic_filter*2), 3, strides=2, padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(dic_dropout),
            layers.Conv3D(int(dic_filter*3), 3, strides=2, padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(dic_dropout),
            layers.Flatten(),
            layers.Dense(1)
        ])
    
    def _make_model(self, enc_filter, gen_filter, dec_filter, dic_filter,
                    enc_dropout, dic_dropout, latent_dim):
        self.E1 = self._make_encoder(self.input_shape, enc_filter, enc_dropout,
                                     latent_dim, pool_size=3)
        self.G1 = self._make_generator(gen_filter)
        self.E2 = self._make_encoder(self.dicrim_shape, enc_filter, enc_dropout,
                                     latent_dim, pool_size=2)
        #self.G2 = self._make_generator(gen_filter)
        self.De = self._make_decoder(dec_filter)
        self.Di = self._make_dicriminator(dic_filter, dic_dropout)
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def encode(self, x):
        mean, logvar = tf.split(x, num_or_size_splits=2, axis=1)
        return mean, logvar
    
    def decode(self, z, apply_sigmoid=False):
        logits = self.De(z)
        if apply_sigmoid:
          probs = tf.sigmoid(logits)
          return probs
        return logits