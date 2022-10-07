import utilities
import numpy
from tcn import TCN
import time
import tensorflow
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
import tensorflow.keras as keras
import pandas

class TCNAE:
    """
    A class used to represent the Temporal Convolutional Autoencoder (TCN-AE).
    ...
    Attributes
    ----------
    model : xxtypexx
        The TCN-AE model.
    Methods
    -------
    build_model(verbose = 1)
        Builds the model
    """
    
    model = None
    
    def __init__(self,
                 ts_dimension = 1,
                 dilations = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
                 nb_filters = 10,
                 kernel_size = 10,
                 nb_stacks = 1,
                 padding = 'same',
                 dropout_rate = 0.0,
                 filters_conv1d = 32,
                 activation_conv1d = 'tanh',
                 latent_sample_rate = 2,
                 pooler = AveragePooling1D,
                 lr = 0.001,
                 conv_kernel_init = 'glorot_normal',
                 loss = 'mse',
                 use_early_stopping = False,
                 error_window_length = 128,
                 verbose = 1,
                 tcn_activation = 'tanh',
                 dense_activation = 'linear'
                ):
        """
        Parameters
        ----------
        ts_dimension : int
            The dimension of the time series (default is 1)
        dilations : tuple
            The dilation rates used in the TCN-AE model (default is (1, 2, 4, 8, 16))
        nb_filters : int
            The number of filters used in the dilated convolutional layers. All dilated conv. layers use the same number of filters (default is 20)
        """
        
        self.ts_dimension = ts_dimension
        self.dilations = dilations
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_stacks = nb_stacks
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.filters_conv1d = filters_conv1d
        self.activation_conv1d = activation_conv1d
        self.latent_sample_rate = latent_sample_rate
        self.pooler = pooler
        self.lr = lr
        self.conv_kernel_init = conv_kernel_init
        self.loss = loss
        self.use_early_stopping = use_early_stopping
        self.error_window_length = error_window_length
        self.tcn_activation = tcn_activation
        self.dense_activation = dense_activation
        
        # build the model
        self.build_model(verbose = verbose)
        
    
    def build_model(self, verbose = 1):
        """Builds the TCN-AE model.
        If the argument `verbose` isn't passed in, the default verbosity level is used.
        Parameters
        ----------
        verbose : str, optional
            The verbosity level (default is 1)
            
        Returns
        -------
        KerasXYZType
        Todo
        Raises
        ------
        NotImplementedError
            If ...
        """
        
        tensorflow.keras.backend.clear_session()
        sampling_factor = self.latent_sample_rate
        i = Input(batch_shape=(None, None, self.ts_dimension))


        # Put signal through TCN. Output-shape: (batch,sequence length, nb_filters)
        tcn_enc = TCN(nb_filters=self.nb_filters, kernel_size=self.kernel_size, nb_stacks=self.nb_stacks, dilations=self.dilations, 
                      padding=self.padding, use_skip_connections=True, dropout_rate=self.dropout_rate, return_sequences=True,
                      kernel_initializer=self.conv_kernel_init, activation=self.tcn_activation, name='tcn-enc')(i)

        # Now, adjust the number of channels...
        enc_flat = Conv1D(filters=self.filters_conv1d, kernel_size=1, activation=self.activation_conv1d, padding=self.padding)(tcn_enc)

        ## Do some average (max) pooling to get a compressed representation of the time series (e.g. a sequence of length 8)
        enc_pooled = self.pooler(pool_size=sampling_factor, strides=None, padding='valid', data_format='channels_last')(enc_flat)
        
        # If you want, maybe put the pooled values through a non-linear Activation
        enc_out = Activation("tanh")(enc_pooled)

        # Now we should have a short sequence, which we will upsample again and then try to reconstruct the original series
        dec_upsample = UpSampling1D(size=sampling_factor)(enc_out)

        #dec_reconstructed = TCN(nb_filters=self.nb_filters, kernel_size=self.kernel_size, nb_stacks=self.nb_stacks, dilations=self.dilations, 
        #                        padding=self.padding, use_skip_connections=True, dropout_rate=self.dropout_rate, return_sequences=True,
        #                        kernel_initializer=self.conv_kernel_init, activation=self.tcn_activation, name='tcn-dec')(dec_upsample)

        # Put the filter-outputs through a dense layer finally, to get the reconstructed signal
        #o = Dense(self.ts_dimension, activation=self.dense_activation)(dec_reconstructed)
        o = Dense(self.ts_dimension, activation=self.dense_activation)(dec_upsample)
        #o = enc_out

        model = Model(inputs=[i], outputs=[o])

        adam = optimizers.Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True)
        model.compile(loss=self.loss, optimizer=adam, metrics=[self.loss])
        if verbose > 1:
            model.summary()
        self.model = model
    
    def fit(self, train_X, train_Y, batch_size=32, epochs=40, verbose = 1,my_callbacks=None):
        #my_callbacks = None
        if self.use_early_stopping:
            my_callbacks = [EarlyStopping(monitor='val_loss', patience=2, min_delta=1e-4, restore_best_weights=True)]

        keras_verbose = 0
        if verbose > 0:
            print("> Starting the Training...")
            keras_verbose = 2
        start = time.time()
        history = self.model.fit(train_X, train_Y, 
                            batch_size=batch_size, 
                            epochs=epochs, 
                            validation_split=0.1, 
                            shuffle=True,
                            callbacks=my_callbacks,
                            verbose=keras_verbose)
        if verbose > 0:
            print("> Training Time :", round(time.time() - start), "seconds.")
        return history

    
    def predict(self, test_X):
        X_rec =  self.model.predict(test_X)
        
        # do some padding in the end, since not necessarily the whole time series is reconstructed
        X_rec = numpy.pad(X_rec, ((0,0),(0, test_X.shape[1] - X_rec.shape[1] ), (0,0)), 'constant') 
        E_rec = (X_rec - test_X).squeeze()
        Err = utilities.slide_window(pandas.DataFrame(E_rec), self.error_window_length, verbose = 0)
        Err = Err.reshape(-1, Err.shape[-1]*Err.shape[-2])
        sel = numpy.random.choice(range(Err.shape[0]),int(Err.shape[0]*0.98))
        mu = numpy.mean(Err[sel], axis=0)
        cov = numpy.cov(Err[sel], rowvar = False)
        sq_mahalanobis = utilities.mahalanobis_distance(X=Err[:], cov=cov, mu=mu)
        # moving average over mahalanobis distance. Only slightly smooths the signal
        anomaly_score = numpy.convolve(sq_mahalanobis, numpy.ones((50,))/50, mode='same')
        anomaly_score = numpy.sqrt(anomaly_score)
        return anomaly_score