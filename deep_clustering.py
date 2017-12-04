import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import copy
import librosa
# import lws

import pickle
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Dense, Reshape, LSTM, Bidirectional, Dropout, GaussianNoise, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import Callback, ModelCheckpoint


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
        self.learning_rate = 0.0001
        self.momentum = 0.9
        self.training_steps = 5000
        self.batch_size = 32


        # Network Parameters
        self.num_input = 129 # bins for spectrogram data input
        self.frame_size = 100 # samples of audio in one frame (training example)
        self.timesteps = 100 # timesteps
        self.num_hidden = 600 # hidden layer num of LSTM cells per layer
        self.input_dropout_rate = 0
        self.recurrent_dropout_rate = 0.
        self.output_dropout_rate = 0.
        self.embedding_size = 40

        # Audio Parameters
        self.sample_rate = 8000
        self.window_size = 256
        self.hop_length = 64

        # try different optimizer, visualize embeddings, visualize masks, mean and stdev of weights


        self.audio_clips = []




    def load_data(self, dir_path):
        print("loading data...")

        for root, directories, filenames in os.walk(dir_path):
            for filename in filenames: 
                # if filename.lower() == ('vocals.wav'):
                if True:
                    # try to load file into an AudioSignal obj, and add it to audio_clips if successful
                    try:
                        print("--loading file...")
                        signal, sr = librosa.load(os.path.join(root,filename), mono=True, sr=44100)
                        # signal = nussl.AudioSignal(os.path.join(root,filename))
                    except:
                        print(filename + " could not be loaded. This file may be in an improper format, or corrupted.")
                        continue

                    # resample audio to 8k and add an AudioSignal obj to our list
                    # change this to nussl
                    signal = librosa.resample(signal, 44100, 8000)
                    # signal = nussl.AudioSignal(audio_data_array=raw_audio)
                    # signal.resample(8000)
                    self.audio_clips.append(signal)
                    print("--file loaded.")

        print("done loading data.")


    def prepare_data(self):
        '''
        This will eventually prepare all the data, right now it just prepares training examples with a two-file mixture
        '''
        # stft_parameters = nussl.StftParams(sample_rate=self.sample_rate, window_length=self.window_size, hop_length=self.hop_length, n_fft_bins=self.num_input)


        print("preparing data...")

        chunks = []

        for signal in self.audio_clips:
            # get stft data

            # signal.stft_params = stft_parameters
            # signal.stft()
            # stft = signal.stft_data
            # print stft
            # get raw audio, reshape to 1D
            raw_audio = signal
            # raw_audio = np.reshape(raw_audio, raw_audio.shape[1])
            # raw_audio = signal.get_channel(0)


            # get log-magnitude stft and pad for chunking
            print("--calculating the STFT...")

            stft = librosa.stft(raw_audio, n_fft=self.window_size, hop_length=self.hop_length)
            stft = librosa.amplitude_to_db(stft)
            pad_length = self.frame_size - (stft.shape[1] % self.frame_size)
            stft = np.lib.pad(stft, ((0, 0), (0, pad_length)), 'constant')

            # reshape chop into chunks of size frame_size
            stft = np.reshape(np.transpose(stft), (-1, self.frame_size, self.num_input))
            chunks.append(stft)

        # add frames together (TODO: generalize this for more than 2 examples)
        length = min([a.shape[0] for a in chunks])

        # fill an array y_true[num_examples, frame_size*num_input, num_speakers] with true mask data (Fatemeh: use embd_size instead of num_speakers for padding)
        # fill an array x[num_examples, frame_size, num_input] with training data
        print("--calculating mask and creating mixture...")

        y_true = np.zeros((length, self.frame_size*self.num_input, self.embedding_size))
        x = np.zeros((length, self.frame_size, self.num_input), dtype=np.complex64)

        # store true labels for some reason maybe we need it later
        y_labels = np.zeros((length*self.frame_size*self.num_input))
        label_counter = 0

        for i in range(0, length):
            for j in range(0, self.frame_size):
                for k in range(0, self.num_input):

                    bin0 = chunks[0][i, j, k]
                    bin1 = chunks[1][i, j, k]

                    # create mixture in x
                    x[i, j, k] = bin0 + bin1

                    #compare magnitudes of each bin for mask values in y_true
                    # ask Fatemeh about using power spectrogram or other representations to compare bins
                    if np.real(bin1) > np.real(bin0):
                        if np.real(bin1) > -40.0:
                            y_true[i, j*self.num_input + k, 1] = 1
                            y_labels[label_counter] = 1
                        else: 
                            y_true[i, j*self.num_input + k, 2] = 1
                            y_labels[label_counter] = 2


                    else:
                        if np.real(bin0) > -40.0:
                            y_true[i, j*self.num_input + k, 0] = 1
                            y_labels[label_counter] = 0
                        else: 
                            y_true[i, j*self.num_input + k, 2] = 1
                            y_labels[label_counter] = 2



                    label_counter += 1


        scaled_length_for_batches = y_true.shape[0]-(y_true.shape[0]%self.batch_size)
        scaled_length_for_labels = scaled_length_for_batches*self.frame_size*self.num_input
        scaled_batch_size = self.batch_size*self.frame_size*self.num_input

        # store in obj, scaled for proper batch size (TODO: how do I not have to do this?)
        # TEMP: get only the last batch, we're trying to overfit
        self.y_true = y_true[scaled_length_for_batches-self.batch_size : scaled_length_for_batches, :, :]
        self.x = x[scaled_length_for_batches-self.batch_size : scaled_length_for_batches, :, :]
        self.y_labels = y_labels[scaled_length_for_labels-scaled_batch_size : scaled_length_for_labels] # y_labels[:self.x.shape[0]*self.x.shape[1]*self.x.shape[2]]

        print("--y_true: ")
        print(self.y_true)
        print("--y_shape: ")
        print(self.y_true.shape)

        # # testing to make sure mixture is going into NN properly
        # stft = np.reshape(self.x, (-1, self.num_input))
        # stft = np.transpose(stft)
        # x_out = librosa.istft(stft, hop_length=self.hop_length)
        # librosa.output.write_wav("mixture_small.wav", x_out, 8000)


        print("done preparing data.")

        dump_obj = True

        if dump_obj:
            print("dumping object to file...")
            pickle.dump(self, open("dc_obj", 'wb'))
            print("object dumped to file.")



    def loss_function_test(self):
        sess = tf.InteractiveSession()

        # grab one batch from the true labels
        y_true = self.y_true[0:32]

        # add random gaussian noise 
        y_pred = y_true + np.random.ranf(y_true.shape)/10

        # compute the dc loss
        loss = self.deep_clustering(y_true, y_pred)
        print("Loss for y_true = y_pred:")
        print(loss.eval(session=sess))


    def train_model(self):

        class WeightNoiseCallback(Callback):

            def on_batch_begin(self, batch, logs={}):
                for i in range(0, 6):
                    layer = self.model.get_layer(index=i)
                    weights = layer.get_weights()
                    for weight_array in weights:
                        weight_array += np.random.normal(loc=0.0, scale=0.01, size=weight_array.shape)
                    
                    layer.set_weights(weights)

        # initialize callbacks
        weight_noise = WeightNoiseCallback() # not working right now, Fatemeh said she didn't even use weight noise
        model_checkpoint = ModelCheckpoint('weights/sigmoid_fixed_threshold/sigmoid-small-lolr.{epoch:02d}-{loss:.2f}.hdf5', monitor='val_loss', save_weights_only=True, mode='auto', period=100)

        print("training model... start praying...")

        self.model.fit(self.x, self.y_true, batch_size=self.batch_size, epochs=self.training_steps, verbose=1, callbacks=[model_checkpoint])

        print("saving trained model weights...")
        self.model.save_weights("deep_clustering_weights_DAPS_sigmoid_fixed_lolr.h5")

        print("model saved.")

        return None



    def predict_from_input(self):
        print("loading weights...")
        self.model.load_weights("deep_clustering_weights.h5")

        print("evaluating input...")
        prediction = self.model.predict(self.x)

        print("prediction: ")
        print(prediction)

        print("prediction mean: {}".format(np.mean(prediction)))



    def create_model(self, training=True): 

        self.model = Sequential()

        if training:
            stateful = False
            batch_size = self.batch_size

        else:  
            stateful = True
            batch_size = 1


        # 1st BLSTM layer with Input and Recurrent Dropout
        # NOTE: will need to call model.reset_states() somehow after each sample
        self.model.add(Bidirectional(LSTM(self.num_hidden/2, activation='tanh',
            return_sequences=True,
            stateful=stateful,
            implementation=1,
            recurrent_dropout=self.recurrent_dropout_rate),
            batch_size=batch_size,
            input_shape=(self.frame_size, self.num_input)))

        # output dropout
        # self.model.add(Dropout(self.output_dropout_rate))

        # 2nd BLSTM layer with Input and Recurrent Dropout
        self.model.add(Bidirectional(LSTM(self.num_hidden/2, activation='tanh',
            return_sequences=True,
            stateful=stateful,
            implementation=1, 
            recurrent_dropout=self.recurrent_dropout_rate),
            batch_size=batch_size))

        # output dropout
        # self.model.add(Dropout(self.output_dropout_rate))

        # feedforward layer
        self.model.add(Dense(self.embedding_size*self.num_input, activation='sigmoid'))

        # We want shape to be: (129, 40) for each timestep
        # Current output shape: (batch_size, num_frames, num_bins*k)
        # Loss Function input shape: (batch_size, num_frames*num_bins, k)
        # V Shape: (batch_size*num_frames*num_bins, k)

        # stochastic gradient descent with momentum
        # MAYBE TRY ADAM with the default values
        # self.optimizer = Adam(lr=self.learning_rate)
        # self.optimizer = SGD(lr=self.learning_rate, momentum=self.momentum)
        self.optimizer = RMSprop(lr=self.learning_rate)

        self.model.compile(optimizer=self.optimizer, loss=self.deep_clustering)

        return self.model


    def print_model(self):
        for i, layer in enumerate(self.model.layers):
            print("layer " + str(i) + ": " + str(layer))
            print("shape: " + str(layer.input_shape) + " --> " + str(layer.output_shape) + '\n')


    def separate(self, audio_path, num_sources=2):

        print("separating audio...")

        # np.set_printoptions(threshold=np.inf)

        try:
            # mixture = AudioSignal(audio_path)
            mixture, sr = librosa.load(audio_path, mono=True, sr=44100)
            mixture = librosa.resample(mixture, 44100, 8000)
        except Exception as e:
            print("invalid file. " + str(e))
            return None

        # convert mixture to 8000hz Mono
        # mixture.to_mono(overwrite=True)
        # TODO: Convert to 8000Hz

        # get the STFT for input into the model
        print("--calculating STFT of mixture...")
        mixture_stft = librosa.stft(mixture, n_fft=self.window_size, hop_length=self.hop_length)
        mixture_input = np.reshape(np.transpose(mixture_stft), (-1, self.frame_size, self.num_input))

        # generate embeddings and flatten for kmeans
        print("--predicting...")
        self.create_model(training=False)
        self.model.load_weights("deep_clustering_weights_DAPS_sigmoid_fixed.h5")
        encoded_mixture = self.model.predict(mixture_input, batch_size=1)
        flattened_encodings = np.reshape(encoded_mixture, (-1, self.frame_size, self.num_input, self.embedding_size))
        flattened_encodings = np.reshape(flattened_encodings, (-1, self.embedding_size))

        print("--clustering...")
        # perform kmeans to obtain class mask
        kmeans = KMeans(n_clusters=num_sources+1, max_iter=10000, n_jobs=-1)
        classes = kmeans.fit_predict(flattened_encodings)
        int_mask = np.reshape(classes, (-1, self.num_input))
        int_mask = np.transpose(int_mask)


        # NEW CLUSTERING ALGORITHM
        # fit_predict for each frame
        # save masks and center values
        # perform clustering among centers
            # use another audio related heuristic to find out which greater cluster each smaller cluster belongs to
        



        # plt.subplot(121)
        print int_mask.shape
        print np.mean(int_mask)
        # plt.imshow(int_mask)

        speaker_stfts = [copy.deepcopy(mixture_stft) for i in range(num_sources+1)]
        ground_truth_stfts = [copy.deepcopy(mixture_stft) for i in range(num_sources+1)]
        speaker_audio = []
        ground_truth_audio = []
        # lws_processor=lws.lws(self.window_size,self.hop_length, mode="speech")

        # form a ground truth mask, so we can mask the mixtures and compare how they sound to our output
        ground_truth_mask = np.reshape(self.y_labels, (-1, self.num_input))
        ground_truth_mask = np.transpose(ground_truth_mask)

        # plt.imshow(ground_truth_mask)
        # plt.subplot(122)

        print("--applying mask...")
        for speaker in range(0, num_sources+1):

            # apply mask for each speaker to the mixture spectrogram
            for i in range(mixture_stft.shape[0]):
                for j in range(mixture_stft.shape[1]):
                    if int_mask[i, j] != speaker:
                        speaker_stfts[speaker][i, j] = 0+0j
                    if ground_truth_mask[i, j] != speaker:
                        ground_truth_stfts[speaker][i, j] = 0+0j

            # reconstruct phase
            # print("--reconstructing phase...")
            # speaker_stfts[speaker] = lws_processor.run_lws(speaker_stfts[speaker])

            # save to file
            print("--saving to file...")
            speaker_audio.append(librosa.istft(speaker_stfts[speaker], hop_length=self.hop_length))
            librosa.output.write_wav("speaker_{}.wav".format(speaker), speaker_audio[speaker], 8000)

            ground_truth_audio.append(librosa.istft(ground_truth_stfts[speaker], hop_length=self.hop_length))
            librosa.output.write_wav("speaker_{}_true.wav".format(speaker), ground_truth_audio[speaker], 8000)

            print("audio files saved.")




        # TODO: Use Mask and SeparationBase class stuff
        # TODO: Figure out what to return: Raw audio? Masks?



    # Deep clustering loss function

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

            # print("Y_Pred Shape: {}".format(y_pred.get_shape()))
            # print("Est_Mask Shape: {}".format(est_mask.get_shape()))

            # weighted true and estimated masks
            true_mask_w = tf.multiply(true_mask,class_weight_tile)
            est_mask_w = tf.multiply(est_mask,class_weight_tile)

            # computation of correlation terms (ask: why do we compute correlation? Is this in place of Frobenius norm?)
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



if __name__ == "__main__":

    dc_obj = DeepClustering()
    dc_obj.load_data('DAPS_examples')
    dc_obj.prepare_data()

    # # dc_obj = pickle.load(open('dc_obj', 'rb'))
    # # dc_obj.loss_function_test()
    dc_obj.create_model()
    dc_obj.print_model()
    dc_obj.train_model()
    # dc_obj.separate("mixture_small.wav", 2)



