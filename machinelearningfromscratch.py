from functools import reduce

import numpy as np
import random
import pickle
import logging



class Model:
    @classmethod
    def save(cls, model, name):
        """Save a model given its name."""
        data = pickle.dumps(model) #data is in binary
        pickle.dump(data, name)
        logging.debug('model saved')

    @classmethod
    def load(cls, name):
        """Load a model given its name."""
        data = pickle.load(name)  #data is in binary
        model = pickle.loads(data)
        logging.debug('model loaded')
        return model

    @classmethod
    def normalize(cls, data, axis=-1):
        return data/np.linalg.norm(data, axis=axis)

    @classmethod
    def random(cls, dims, **kwargs):
        """Return a model with random weights and biases given the dim."""
        return cls(
            [Layer.random(width, height) for (width, height) in zip(dims[:-1], dims[1:])],
            **kwargs
        )

    def __init__(self,
        layers,
        learning_rate = 0.1,
        minibatch_size = 10,
        epochs = 1000,
        loss_function = lambda y_, y: y*np.log(y_) + (1-y)*np.log(1-y_),
        loss_derivative = lambda y_, a: y_-a
    ):
        """Create a neural network model."""
        self.layers = layers
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.epochs = epochs

    def __str__(self):
        layers = ''.join(map(lambda l:f'\n - {str(l)}', self.layers))+'\n'
        name = type(self).__name__
        return f"{name}[{layers}]"

    @property
    def dims(self):
        return [self.layers[0].width] + [l.height for l in self.layers]

    @property
    def depth(self):
        return len(self.dims)

    def cost(self):
        return np.mean()

    def train(self, training_data):
        """Train the model given some training data."""
        m = self.minibatch_size
        for i in range(self.epochs):
            random.shuffle(training_data)
            minibatches = [training_data[i*m: (i+1)*m] for i in range(len(training_data)//m)]
            for minibatch in minibatches:
                self.train_minibatch(minibatch)

    def train_minibatch(self, minibatch):
        """Train the model on a minibatch."""
        # grosse refléction ici
        nabla_w, nabla_b = zip(np.array(
            [(np.zeros(l.bias.shape), np.zeros(l.weight.shape)) for l in self.layers]
        ))
        for (x, y) in minibatch:
            delta_nabla_w, delta_nabla_b = self.back_propagation(x, y)
            nabla_w += delta_nabla_w # operations on numpy arrays
            nabla_b += delta_nabla_b
        for layer, nw, nb in zip(self.layers, nabla_w, nabla_b):
            layer.weight -= nw*learning_rate/len(minibatch)
            layer.bias -= nb*learning_rate/len(minibatch)

    def back_propagation(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_w, nabla_b = zip(np.array(
            [(np.zeros(l.bias.shape), np.zeros(l.weight.shape)) for l in self.layers]
        ))
        # feedforward
        a = x
        # self.activations = [a] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for layer in self.layers:
            z = np.dot(layer.weight, a)+layer.biases
            zs.append(z)
            a = layer.activation_function(z)
            layer.activations = a
            # self.activations.append(a)

        backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.depth):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)            
            
    def predict(self, x):
        """Predict a result given an input."""
        y = x
        for layer in self.layers:
            y = layer.predict(y)
        return y

    __call__ = predict


class Classifier(Model):
    def evaluate(self, test_data):
        """Returm the percentage of correct predictions
        given the test data."""
        return sum(int(np.argmax(self(x)) == y) for x, y in test_data)/len(test_data)


class Layer:
    @classmethod
    def random(cls, width, height, **kwargs):
        return cls(
            np.random.rand(height, width),
            np.random.rand(height,1),
            **kwargs
        )

    def __init__(self,
        weight,
        bias,
        activation_function=lambda x:np.max(0,x), # relu
        activation_derivative=lambda x:x>=0,
        activation=[],
    ):
        self.weight = weight
        self.bias = bias
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.activation = activation

    def __str__(self):
        return f"{type(self).__name__}({self.width},{self.height})"

    @property
    def width(self):
        return self.weight.shape[1]

    @property
    def height(self):
        return self.weight.shape[0]

    def predict(self, x):
        return self.activation_function(np.dot(self.weight, x)+self.bias)

def test_layer():
    l = Layer.random(30, 10)
    x = np.random.rand(30, 1)
    y = l.predict(x)
    print(y)

def test_reddit():
    import sqlite3
    conn = sqlite3.connect('reddit')
    c = conn.cursor()
    c.execute('select * from reddit')
    results = c.fetchmany(10)
    contents = [r[5] for r in results]
    training_data = list(zip(contents[:-1], contents[1:]))
    print(training_data)
    model = Model.random([])
    model.train(results)

def test_mnist():
    import tensorflow as tf
    import tensorflow_datasets as tfds
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    print(ds_train)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )

    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )
    # print(ds_train['shapes'])
    # print(ds_train.__dict__)
    # print(ds_train.shape)


def test_mnist2():
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(training_data.shape)

if __name__ == "__main__":
    test_mnist()