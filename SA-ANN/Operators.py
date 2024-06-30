import tensorflow as tf

def create_ann_model(num_layers, units_per_layer, activation_per_layer, optimizer, loss, input_shape):
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=units_per_layer[0], activation=activation_per_layer[0], input_shape=(input_shape,)))
    for i in range(1, num_layers):
        ann.add(tf.keras.layers.Dense(units=units_per_layer[i], activation=activation_per_layer[i]))
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    ann.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return ann
