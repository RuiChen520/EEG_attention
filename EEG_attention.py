import scipy.io as scio
import time as mytime
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import numpy as np
from utils import *

dataFile = './dataset/dataset_BCIcomp1.mat'
data = scio.loadmat(dataFile)

print(data.keys())
print(np.array(data['x_train']).shape)
print(np.array(data['x_test']).shape)
print(np.array(data['y_train']).shape)
print(data['y_train'][0])

f = 128
time = 3
start = time * f
Ts = 1152
total_num = 140
m = 100
test_num = 40
Tx = 1152 - time * f  # 1152 - 3*128 = 768
Ty = 1
channels = 3
classes = 2
X = np.array(data['x_train'])[:][:][start:]
X = X.transpose(2, 0, 1)
Y = np.zeros((total_num, 1, 2))
for i in range(total_num):
    Y[i] = np.array(list(map(lambda x: to_categorical(x, num_classes=2), data['y_train'][i]-1)))
# 划分训练集和测试集
X_train = X[0:m][:][:]
X_test = X[m:][:][:]
Y_train = Y[0:m][:][:]
Y_test = Y[m:][:][:]
print('origin:', data['y_train'])
print('after map to one-hot:', Y_train)
print(X_train.shape)
print(Y_train.shape)

# Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)


# GRADED FUNCTION: one_step_attention
def one_step_attention(a, s_prev):
    """
    :param a:
    :param s_prev:
    :return:
    """
    s_prev = repeator(s_prev)

    concat = concatenator([a, s_prev])

    e = densor1(concat)

    energies = densor2(e)

    alphas = activator(energies)

    context = dotor([alphas, a])
    return context


n_a = 32
n_s = 64
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(classes, activation=softmax)


# GRADED FUNCTION: model

def model(Tx, Ty, n_a, n_s, channels, classes):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"
    Returns:
    model -- Keras model instance
    X: Tensor("input_3:0", shape=(?, 30, 37), dtype=float32)
    s0 Tensor("s0_2:0", shape=(?, 64), dtype=float32)
    c0 Tensor("c0_2:0", shape=(?, 64), dtype=float32)
    s Tensor("s0_2:0", shape=(?, 64), dtype=float32)
    c Tensor("c0_2:0", shape=(?, 64), dtype=float32)
    """

    X = Input(shape=(Tx, channels))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    # Initialize empty list of outputs
    outputs = []

    ### START CODE HERE ###

    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)

    # Step 2: Iterate for Ty steps
    for t in range(Ty):
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention(a, s)

        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)

        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)

    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs=[X, s0, c0], outputs=outputs)

    ### END CODE HERE ###

    return model


model = model(Tx, Ty, n_a, n_s, channels, classes)


model.summary()

# START CODE HERE ### (≈2 lines)

opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Y_train.swapaxes(0,1))
o1 = np.array(outputs)

print(o1.shape)
print(X_train.shape)

time_start = mytime.time()
model.fit([X_train, s0, c0], outputs, epochs=2000, batch_size=100)
time_end = mytime.time()
print('time cost',time_end-time_start,'s')
model.save_weights('model.h5')

sum = 0
print('X_test:', X_test.shape)
print('Y_test:', Y_test.shape)
for i in range(test_num):
    test = np.expand_dims(X_test[i][:][:], axis=0)
    prediction = model.predict([test, s0, c0])
    prediction = np.argmax(prediction, axis=-1) + 1
    label = np.argmax(Y_test[i], axis=-1) + 1
    print(prediction, label)
    if prediction == label:
        sum += 1

print('result: ', sum*1.0/test_num)