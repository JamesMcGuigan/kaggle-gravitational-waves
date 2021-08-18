import tensorflow as tf
from tensorflow.keras import layers


def build_model():
    inputs = layers.Input(shape=(27, 128))

    gru1  = layers.Bidirectional(layers.GRU(128, return_sequences=True), name='gru_1')
    gru2  = layers.Bidirectional(layers.GRU(128, return_sequences=True), name='gru_2')
    pool1 = layers.GlobalAveragePooling1D(name='avg_pool')
    pool2 = layers.GlobalMaxPooling1D(name='max_pool')

    x = gru1(inputs)
    x = gru2(x)
    x = tf.keras.layers.Concatenate()([pool1(x), pool2(x)])

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(1,   activation="sigmoid", name="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model
