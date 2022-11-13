import numpy as np
import tensorflow as tf
from keras.layers import Layer, Dense, Flatten, BatchNormalization
from keras import regularizers
import plotly.express as px
import plotly.graph_objects as go
import time
from scipy import special
import itertools

k = 8 # number of bits per symbol
N = 16 # number of bits per coded symbol

bits_8 = np.array([list(i) for i in itertools.product([0, 1], repeat=8)])

G = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
              [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
              [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=bool) # Generatice matrix

train_losses = []
val_losses = []
test_losses = []

train_bers = []
val_bers = []
test_bers = []

train_epoch_losses = []
train_epoch_blers = []

test_epoch_losses = []
test_epoch_blers = []


def to_8(signal):

    sign_mat = signal.reshape(int(len(signal)/8), 8)
    return sign_mat


def to_16(signal):

    signal = np.dot(signal, G) % 2
    return signal


def total_errors(y_true, y_pred):
    errors = tf.not_equal(y_true, tf.round(y_pred))
    sum = tf.reduce_sum(tf.cast(errors, tf.float32))
    return sum


def bler(y_true, y_pred):
    errors = tf.not_equal(y_true, tf.round(y_pred))
    bler = tf.reduce_mean(tf.cast(errors, tf.float32))
    return bler


def random_signal(simSize):
    return np.random.randint(low=0, high=2, size=simSize)


def construct_batch(codebook, batch_size, codebook_8):

    batch_indices = np.random.choice(len(codebook), batch_size)
    batch_data = codebook[batch_indices]
    batch_data = tf.convert_to_tensor(batch_data, dtype=tf.float32)

    original_data_batch = codebook_8[batch_indices]
    original_data_batch = tf.convert_to_tensor(original_data_batch, dtype=tf.float32)

    return batch_data, original_data_batch


def BPSK_mod(bitstream, M ,amplitude):
    """
    The BPSK_mod function takes in a bitstream and outputs the modulated signal.
    It uses the formula for BPSK modulation, which is:
        s(t) = A*cos(2pi*f_c*t + phi), where f_c is the carrier frequency, A is amplitude of cosine wave,
        and phi is phase shift.

    :param bitstream: Used to Represent the bitstream to be transmitted.
    :param M: Used to Define the number of constellation points.
    :param amplitude: Used to Set the amplitude of the output signal.
    :return: The modulated bits.
    """

    m = np.arange(0, M)  # every possible inputs

    constellation = amplitude*np.cos((m/M)*2*np.pi)  # modulation formula for every bit in stream
    mod_bits = constellation[bitstream]

    return mod_bits



def BPSK_thresh(modulated_signal, EbNo_db):
    """
    The BPSK_thresh function takes in a modulated signal vector and an array of EbNo values.
    It then adds noise to the modulated signal using the N0 value corresponding to each EbNo value,
    and sets a threshold at 0. It returns an array of binary vectors representing the recieved signals.

    :param modulated_signal: Used to Pass the modulated signal to the function.
    :param EbNo_db: Used to Determine the noise variance.
    :return: A matrix of the same size as the input, but with each element being either a 0 or 1.
    """

    thresholded_sign = []

    for index, EbNo in enumerate(EbNo_db):

        EbNo_linear = 10**(EbNo/10)  #log to linear scale of EbNo
        power = sum(abs(modulated_signal)**2)/len(modulated_signal)  # power of one-dim (binary) modulated signal vector
        N0 = power/(2 * EbNo_linear ) # noise spectral density function
        noise = np.sqrt(N0)*np.random.standard_normal(modulated_signal.shape)  # defining natural noise in fct of sigma=sqrt(N0/2)

        recieved_signal = modulated_signal + noise
        detected_signal = (recieved_signal <= 0).astype(int)  # setting threshold at 0
        thresholded_sign.append(detected_signal)

    thresholded_sign = np.asarray(thresholded_sign)

    return thresholded_sign


def BPSK_theory(EbNo_db):
    """
    The BPSK_theory function computes the theoretical bit error rate (BER) for BPSK modulation.
    It takes in a vector of EbN0 values and returns a vector of corresponding BER values.
    The function uses the formula: P(e) = 0.5 - 0.5*erf(sqrt(2)*sqrt(SNR)/2).

    :return: The theoretical value of the ber for a binary symmetric channel.
    """
    EbN0_linear = 10**(EbNo_db/10)
    value = np.sqrt(2*EbN0_linear)
    prob_ber_theory = 0.5-0.5*special.erf(value/np.sqrt(2))

    return prob_ber_theory


def bit_error_rate(signal, original):
    """
    The bit_error_rate function takes in a signal and the original signal.
    It compares the two and returns the number of errors.
    """

    ber = np.sum(signal != original)/len(signal)
    return ber


def min_square_distance(codeword, bits_16):
    """
    The min_square_distance function takes in a codeword and the list of 16 bit binary words.
    It then calculates the distance between each word and the codeword, returning
    the minimum of the list of distances.
    """

    distances = []
    codeword = np.asarray(codeword)
    bits_16 = np.asarray(bits_16)
    for i in range(bits_16.shape[0] - 1):
        distances.append(np.linalg.norm(np.subtract(bits_16[i], codeword)))
    min_index = np.argmin(distances)
    return min_index


def awgn(signal_mat, EbN0):

    """
    The awgn function adds noise to a signal matrix.
    It takes two arguments: singal_mat and EbN0.
    The function returns the noisy signal.

    :param singal_mat: Define the singal matrix
    :param EbN0: Define the noise variance
    :return: The noise matrix
    """

    EbNo_linear = 10**(EbN0/10) #log to linear scale of EbNo
    power = np.sum(np.sum(abs(signal_mat)**2))/signal_mat.size # power of multi-dim (binary) modulated signal matrix
    noise_variance = power/(EbNo_linear) # s^2=N0/2
    noise = np.sqrt(noise_variance/2) * np.random.standard_normal(signal_mat.shape) # defining natural noise in fct of sigma=sqrt(N0/2)
    noisy_signal = signal_mat + noise # recieved signal

    return noisy_signal

def BPSK_MAP_decoder(modulated_signal, bits_16, simSize, snr):

    """ The BPSK_MAP_decoder function takes in a modulated signal and the 16-bit dictionary.
    It then calculates the distance between each word and the modulated signal, returning
    the minimum of the list of distances.

    :param modulated_signal: Define the modulated signal
    :param bits_16: Define the 16-bit dictionary
    :return: The decoded signal
    """

    # print(f"\n- Bitstream size : {simSize}\n- EbN0 max : {np.max(EbNo_db)}")
    BER_ = 0
    times = 0
    n_var = snr
    start_ebno = time.time()
    ber = 0
    noisy_signal = awgn(modulated_signal, n_var) # add noise to signal matrix using EbNo value
    print(f"\nComputing BER for EbN0 = {n_var}dB")

    for index, signal in enumerate(noisy_signal):

        min_distance_index = min_square_distance(signal, bits_16) # finding the closest codeword
        bitstream_8_decoded = bits_8[min_distance_index] # decoding the 8-bit codeword
        ber += bit_error_rate(bitstream_8_decoded, test_book[index]) # calculating the BER

    BER_ = ber / (modulated_signal.shape[0]) # calculating the BER for each EbNo value
    end_ebno = time.time()
    times = end_ebno - start_ebno
    print("EbNo : ", n_var, "\tBER : ", BER_, "\t Time :", times)
    return BER_



class Modulating(Layer):
    """
    The Modulating class is used to modulate the signal.
    """
    def __init__(self):
        super(Modulating, self).__init__()

    def call(self, x):
        m = tf.range(0, 2)
        constellation = tf.cos((m/2)*2*np.pi)
        mod_bits = tf.gather(constellation, tf.cast(x, tf.int32))
        return mod_bits



class AGWNoise(Layer):
    """
    The AGWNoise class is used to add AWGN noise to the signal.
    """
    def __init__(self, EbNo):
        super(AGWNoise, self).__init__()
        self.EbNo_db = tf.Variable(EbNo, trainable=False)
        self.EbNo_linear = tf.pow(10, self.EbNo_db / 10) # log to linear scale of EbNo
        self.noise_var = tf.sqrt(1 / (2*self.EbNo_linear)) # noise variance

    def call(self, x):

        x = tf.cast(x, tf.float32)
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=self.noise_var)
        noisy_bits = x + noise

        return noisy_bits



class AbstractChannel(Layer):
    """
    The AbstractChannel class is used to simulate the channel before decoding.
    It modulates and adds AWGN noise to the signal.
    """
    def __init__(self, EbNo):
        super(AbstractChannel, self).__init__()
        self.EbNo_db = EbNo
        self.modulated_bits = Modulating()
        self.noisy_bits = AGWNoise(EbNo=self.EbNo_db)
        self.loss = tf.keras.losses.MeanSquaredError()

    def build(self, input_shape):
        super(AbstractChannel, self).build(input_shape)
        self.trainable = False

    def call(self, x):
        modulated_bits = self.modulated_bits(x)
        self.add_loss(self.loss)

        return self.noisy_bits(modulated_bits)



class AbsoluteErrors(tf.keras.metrics.Metric):
    def __init__(self, name='absolute_errors', **kwargs):
        super().__init__(name=name, **kwargs)
        self.errors_sum = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        errors = tf.not_equal(y_true, tf.round(y_pred))
        sum = tf.reduce_sum(tf.cast(errors, tf.float32))
        self.errors_sum.assign_add(sum)

    def result(self):
        return self.errors_sum



class MAPDecoder(Layer):
    """
    The MAPDecoder class is used to decode the signal with a 3 layer network.
    """
    def __init__(self, batch_size, original_dim, layers, output_dim):
        super(MAPDecoder, self).__init__()

        self.flatten_layer = Flatten()
        self.input_layer = Dense(units=layers[0], activation='relu', input_shape=(batch_size, original_dim), trainable=True)

        self.hidden_layer_1 = Dense(units=layers[1], activation='relu', trainable=True)
        self.hidden_layer_2 = Dense(units=layers[2], activation='relu', trainable=True)
        self.output_layer = Dense(units=output_dim, activation='sigmoid', trainable=True)
        self.batch_norm = BatchNormalization()

        self.error_fcn = AbsoluteErrors()
        self.loss = tf.keras.losses.MeanSquaredError()

    def call(self, x, y, training):

        x = self.flatten_layer(x)
        x = self.input_layer(x)
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)

        x = self.batch_norm(x)

        x = self.output_layer(x)

        errs = self.error_fcn(y, x)
        self.add_metric(errs, name="errors")
        self.add_loss(self.loss)

        return x



class ChannelCoding(tf.keras.Model):
        """
        The ChannelCoding class is used to encode the signal by passing it through the channel, and then decode it with the decoder.
        """
        def __init__(self, batch_size, EbNo, layers):
            super(ChannelCoding, self).__init__()

            self.channel = AbstractChannel(EbNo=EbNo)
            self.decoder = MAPDecoder(batch_size=batch_size, original_dim=16, output_dim=8, layers=layers)

        def call(self, x, y, training):

            x = self.channel(x)
            x = self.decoder(x, y, training)

            return x


prop = 1 # proportion of training data
simSize_train = int((2**k) * prop) # number of bits to send per epoch of training
simSize_test = 2*10**5 # number of bits to send just for testing

bits_16 = to_16(bits_8) # convert 8-bit dictionary to 16-bit dictionary
batch_size = 255 # batch size
epochs = 2**16 # number of epochs
learning_rate = 1e-3

network_arch_1 = [128, 64, 32]
network_arch_2 = [256, 128, 64]
network_arch_3 = [512, 256, 128]

channel_1 = ChannelCoding(batch_size=batch_size, EbNo=1.0, layers=network_arch_1)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001, decay_steps=(simSize_train/batch_size)*100, decay_rate=1, staircase=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

tf.function
def training_step(channel, x, y):
    with tf.GradientTape() as tape:
        logits = channel(x, y, training=True)
        loss = mse_loss_fn(y, logits)
    gradients = tape.gradient(loss, channel.trainable_weights)
    optimizer.apply_gradients(zip(gradients, channel.trainable_weights))
    metric = bler(y, logits)
    return loss, metric

def testing_step(channel, x, y):
    logits = channel(x, y, training=False)
    loss = mse_loss_fn(y, logits)
    metric = bler(y, logits)
    return loss, metric

# Training loop

print("Training with parameters:")
print("Arch", network_arch_1)
print("Number of epochs:", epochs)
print("Batch size:", batch_size)
print("Learning rate:", learning_rate)
print("Propotion of training data:", prop)

for epoch in range(epochs):
    train_book = bits_8[0:simSize_train]
    train_data = to_16(train_book)

    for i in range(int(len(train_data)/batch_size)):
        start = time.time()
        batch, original_batch_data = construct_batch(train_data, batch_size, train_book)
        loss, metric = training_step(channel_1, batch, original_batch_data)
        train_losses.append(loss.numpy())
        train_bers.append(metric.numpy())

    stop = time.time()

    epoch_loss = np.sum(train_losses)/len(train_losses)
    train_epoch_losses.append(epoch_loss)

    epoch_bler = np.sum(train_bers)/len(train_bers)
    train_epoch_blers.append(epoch_bler)

    print("TRAIN --- Epoch:", epoch, "\t| Loss:", epoch_loss, "\t| BLER:", epoch_bler, "\t| Time:", stop-start)


print(channel_1.summary())

model_dir = './model2'
model_name = 'channel_coding_model_128_64_32_bn'

channel_1.save_weights(model_dir + '/' + model_name)

mean_test_bler = []
map_ber = []
# Testing loop

batch_size = 10000

test_book = to_8(random_signal(simSize=simSize_test))
test_data = to_16(test_book)

EbNo_db = tf.range(0, 10, 1, dtype=tf.float32)
test_models = [ChannelCoding(batch_size=batch_size, EbNo=EbNo_db[i], layers=network_arch_1) for i in range(len(EbNo_db))]
total_bers = []

computed_test_ber = []

n_batches = int(len(test_data)/batch_size)

for snr in range(len(EbNo_db)):
    model_dir = './model2'
    model_name = 'channel_coding_model_128_64_32_bn'
    test_model = test_models[snr]
    test_model.load_weights(model_dir + '/' + model_name )

    bitstream_16 = to_16(test_book) # converting the 8-bit bitstream to 16-bit bitstream
    modulated_signal = BPSK_mod(bitstream_16, 2, 1) # modulating the 16-bit bitstream
    computed_test_ber.append(BPSK_MAP_decoder(bitstream_16, bits_16, simSize_test, snr))

    test_bers = []
    ber = 0
    for i in range(n_batches):
        start_timer = time.time()

        test_batch, original_test_batch_data = construct_batch(test_data, batch_size, test_book)
        loss, metric = testing_step(test_model, test_batch, original_test_batch_data)

        stop_timer = time.time()

        ber += metric.numpy()

        test_losses.append(loss.numpy())
        test_bers.append(metric.numpy())

    epoch_loss = np.sum(test_losses)/len(test_losses)
    test_epoch_losses.append(epoch_loss)

    epoch_bler = np.sum(test_bers)/len(test_bers)
    test_epoch_blers.append(epoch_bler)

    print("TEST --- Epoch:", snr, "\t| Loss:", epoch_loss, "\t| BLER:", epoch_bler, "\t| Time:", stop_timer-start_timer)


    total_bers.append(ber / n_batches) # average bler over all batches
    mean_test_bler.append(np.mean(test_bers)) # mean BER over the entire test set for this SNR


# Showing the training and testing loss
# Showing epoch_blers and losses for train and test
fig_epoch_losses_train = px.line(x=np.arange(len(train_epoch_losses)),
                                 y=train_epoch_losses,
                                 title='Training loss',
                                 labels={'x': 'Epoch', 'y': 'Train Loss'},
                                 color_discrete_sequence=['#1f77b4'])


fig_epoch_losses_test = px.line(x=np.arange(len(test_epoch_losses)),
                                y=test_epoch_losses,
                                title='Testing loss',
                                labels={'x': 'Epoch', 'y': 'Test Loss'},
                                color_discrete_sequence=['#ff7f5e'])

fig_epoch_blers_train = px.line(x=np.arange(len(train_epoch_blers)),
                                y=train_epoch_blers,
                                title='Epoch train BER',
                                labels={'x': 'Epoch', 'y': 'Train BER'},
                                color_discrete_sequence=['#1f77a2'])

fig_epoch_blers_test = px.line(x=np.arange(len(test_epoch_blers)),
                               y=test_epoch_blers,
                               title='Epoch test BER',
                               labels={'x': 'Epoch', 'y': 'Test BER'},
                               color_discrete_sequence=['#ff7f45'])

fig_epoch = go.Figure(data=fig_epoch_losses_train.data + fig_epoch_blers_train.data + fig_epoch_losses_test.data + fig_epoch_blers_test.data)

fig_epoch.update_layout(title_text='BER over epochs')
fig_epoch.update_xaxes(title_text='Number of epochs')
fig_epoch.update_yaxes(title_text='BER')
fig_epoch.update_layout(template='plotly_dark', legend_title_text='Legend', legend_traceorder='reversed')
fig_epoch.show()

fig_test_ber = px.line(x=EbNo_db,
                       y=mean_test_bler,
                       title='BER over EbNo',
                       labels={'x': 'EbNo', 'y': 'BER_DL_Test'},
                       color_discrete_sequence=['#1f77b4'])

fig_map_ber = px.line(x=EbNo_db,
                      y=computed_test_ber,
                      title='BER over EbNo',
                      labels={'x': 'EbNo', 'y': 'BER_MAP (reference)'},
                      color_discrete_sequence=['#ff7f45'])

ber_compare = go.Figure(data=fig_test_ber.data + fig_map_ber.data)
ber_compare.update_yaxes(type='log')
ber_compare.update_layout(template='plotly_dark', legend_title_text='Legend', legend_traceorder='reversed')
ber_compare.show()