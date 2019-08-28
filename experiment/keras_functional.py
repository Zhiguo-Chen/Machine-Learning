from tensorflow.python.keras import Input 
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers
import tensorflow as tf 



main_input = Input(shape=(100,), dtype='int32', name='main_input')
x = layers.Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
lstm_out = layers.LSTM(32)(x)
auxiliary_output = layers.Dense(1 ,activation='sigmoid', name='aux_output')(lstm_out)
aux_input = Input(shape=(5,), name='aux_input')
x = layers.concatenate([lstm_out, aux_input])
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
main_output = layers.Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(inputs=[main_input, aux_input], outputs=[main_output, auxiliary_output])
model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1., 0.2])


