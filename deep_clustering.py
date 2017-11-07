import tensorflow as tf
from sklearn.cluster import KMeans
from nussl import AudioSignal
import os


from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, LSTM, Bidirectional, Dropout, Embedding
from tensorflow.contrib.keras.python.keras.optimizers import SGD


# questions for Fatemeh:
# 600 units total in BLSTM or 600 each for forward/backward?
# should it be stateful and reset per sample? or reset per batch?

# questions for Prem:
# check layer dimensionality
# y_pred missing dimension when sent to deep_clustering (rank 2 instead of rank 3)
# general object-oriented ml stuff (what do I store with the class?)



class DeepClustering():

    def __init__(self):
        # Training Parameters
        self.learning_rate = 0.00001
        self.momentum = 0.9
        self.training_steps = 10000
        self.batch_size = 32


        # Network Parameters
        self.num_input = 129 # bins for spectrogram data input
        self.frame_size = 100 # samples of audio in one frame (training example)
        self.timesteps = 100 # timesteps
        self.num_hidden = 600 # hidden layer num of LSTM cells per layer
        self.input_dropout_rate = 0
        self.recurrent_dropout_rate = 0
        self.output_dropout_rate = 0
        self.embedding_size = 40


        self.audio_clips = []


    def load_data(self, dir_path):
        '''
        Loads raw audio clips into individual AudioSignal objects
        '''
        for root, directories, filenames in os.walk(dir_path):
            for filename in filenames: 
                if filename.lower().endswith('wav'):
                    # try to load file into an AudioSignal obj, and add it to audio_clips if successful
                    try:
                        signal = nussl.AudioSignal(os.path.join(root,filename))
                    except:
                        print(filename + " could not be loaded. This file may be in an improper format, or corrupted.")
                        continue

                    self.audio_clips.append(signal) 



    def train_model(self):
    # window 256, hop 64


        return None

    def create_model(self): 

        self.model = Sequential()


        # 1st BLSTM layer with Input and Recurrent Dropout
        # NOTE: will need to call model.reset_states() somehow after each sample
        self.model.add(Bidirectional(LSTM(self.num_hidden/2, activation='tanh',
            unit_forget_bias=True,
            dropout=self.input_dropout_rate,
            recurrent_dropout=self.recurrent_dropout_rate,
            return_sequences=True,
            stateful=False,
            implementation=1),
            batch_size=self.batch_size,
            input_shape=(self.frame_size, self.num_input)))

        # output dropout
        self.model.add(Dropout(self.output_dropout_rate))

        # 2nd BLSTM layer with Input and Recurrent Dropout
        self.model.add(Bidirectional(LSTM(self.num_hidden/2, activation='tanh',
            unit_forget_bias=True,
            dropout=self.input_dropout_rate,
            recurrent_dropout=self.recurrent_dropout_rate,
            return_sequences=True,
            stateful=False,
            implementation=1),
            batch_size=self.batch_size))

        # output dropout
        self.model.add(Dropout(self.output_dropout_rate))

        # feedforward layer
        self.model.add(Dense(self.embedding_size*self.num_input, activation='softmax'))

        # We want shape to be: (129, 40) for each timestep
        # Current output shape: (batch_size, num_frames, num_bins*k)
        # Loss Function input shape: (batch_size, num_frames*num_bins, k)
        # V Shape: (batch_size*num_frames*num_bins, k)

        # stochastic gradient descent with momentum
        # MAYBE TRY ADAM with the default values
        self.optimizer = SGD(lr=self.learning_rate, momentum=self.momentum)

        self.model.compile(optimizer=self.optimizer, loss=self.deep_clustering)

        return self.model


    def print_model(self):
        for i, layer in enumerate(self.model.layers):
            print("layer " + str(i) + ": " + str(layer))
            print("shape: " + str(layer.input_shape) + " --> " + str(layer.output_shape) + '\n')


    def separate(self, audio_path, num_sources):

        try:
            mixture = AudioSignal(audio_path)
        except:
            print("invalid file.")
            return None

        # convert mixture to 8000hz Mono
        mixture.to_mono(overwrite=True)
        # TODO: Convert to 8000Hz

        # get the STFT for input into the model
        mixture_stft = mixture.stft_data(n_frequency_bins=self.num_input)

        # generate embeddings and flatten for kmeans
        encoded_mixture = self.model.predict(mixture_stft, batch_size=32)
        flattened_encodings = np.reshape(encoded_mixture, (-1, self.embedding_size))

        # perform kmeans to obtain class mask
        kmeans = KMeans(n_clusters=num_sources, random_state=0)
        classes = kmeans.fit_predict(flattened_encodings)
        int_mask = np.reshape(classes, mixture_stft.shape)

        # TODO: Use Mask and SeparationBAse class stuff
        # TODO: Figure out what to return: Raw audio? Masks?






    # Deep clustering loss function

    @classmethod
    def deep_clustering(dc_obj, y_true, y_pred):
        """
        This function implements the deep clustering loss function.

        Input dimensions: [batch_size,num_frames,embedding_size]
        """


        with tf.name_scope('deep_clustering'):

            # dimension of the embedding space
            # embd_dim = y_pred.get_shape().as_list()[2]
            embd_dim = dc_obj.embedding_size

            # form the true mask tensor
            true_mask = tf.cast(tf.reshape(y_true,(-1,embd_dim)),tf.float32)

            # compute the matrix of class weights
            sum_of_true_mask = tf.expand_dims(tf.reduce_sum(true_mask,0),0)
            ones_col = tf.expand_dims(tf.reduce_mean(tf.ones_like(true_mask),1),1)
            tile_of_class_weights = tf.matmul(ones_col,sum_of_true_mask)
            class_weight_mat = tf.multiply(tile_of_class_weights,true_mask)
            class_weight_mat = tf.reduce_sum(class_weight_mat,1)
            class_weight_sqrt = tf.div(tf.constant(1,dtype=tf.float32),tf.sqrt(class_weight_mat))

            # reduce computational complexity by using tiled matrix instead of diagonal
            class_weight_tile = tf.tile(tf.expand_dims(class_weight_sqrt,1),[1,embd_dim])

            # dimensionality reduction for the estimated mask tensor, TODO: make this more efficient
            # don't think I need this unless the reshape is not correct
            # est_mask = tf.cast(tf.reshape(y_pred,(dc_obj.batch_size, dc_obj.num_frames, dc_obj.num_bins, embd_dim)),tf.float32)

            # form the estimated mask tensor
            est_mask = tf.cast(tf.reshape(y_pred,(-1,embd_dim)),tf.float32)


            # print "Y_Pred Shape: {}".format(y_pred.get_shape())
            # print "Est_Mask Shape: {}".format(est_mask.get_shape())

            # weighted true and estimated masks
            true_mask_w = tf.multiply(true_mask,class_weight_tile)
            est_mask_w = tf.multiply(est_mask,class_weight_tile)

            # computation of correlation terms
            true_mask_autocrr = tf.cast(tf.matmul(true_mask_w,true_mask,transpose_a=True),tf.float32)
            est_mask_autocrr = tf.cast(tf.matmul(est_mask_w,est_mask,transpose_a=True),tf.float32)
            true_est_crosscrr = tf.cast(tf.matmul(est_mask_w,true_mask,transpose_a=True),tf.float32)

            # computation of norm-squared terms
            norm_true = tf.cast(tf.trace(tf.matmul(true_mask_autocrr,true_mask_autocrr,transpose_b=True)),tf.float32)
            norm_est = tf.cast(tf.trace(tf.matmul(est_mask_autocrr,est_mask_autocrr,transpose_b=True)),tf.float32)
            norm_true_est = tf.cast(tf.trace(tf.matmul(true_est_crosscrr,true_est_crosscrr,transpose_b=True)),tf.float32)

            # compute the deep clustering loss
            dc_loss = norm_true + norm_est - tf.multiply(tf.constant(2,dtype=tf.float32),norm_true_est)

            # normalize the loss
            norm_term = tf.add(norm_true,norm_est)
            dc_loss_norm = tf.div(dc_loss,norm_term)

            return dc_loss_norm


dc_obj = DeepClustering()
sc_obj.load_data('./DSD100')
dc_obj.create_model()
dc_obj.print_model()

