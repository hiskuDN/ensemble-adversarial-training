import keras
from keras import backend as K
from tensorflow.python.platform import flags
from keras.models import save_model
import tensorflow._api.v2.compat.v1 as tf
from tf_utils import tf_train, tf_test_error_rate
from mnist import *


FLAGS = flags.FLAGS


def main(model_name, model_type):
    tf.disable_v2_behavior()
    
    np.random.seed(0)
    assert keras.backend.backend() == "tensorflow"
    set_mnist_flags()
    

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist()

    data_gen = data_gen_mnist(X_train)

    x = K.placeholder((None,
                       28,
                       28,
                       1
                       ))

    y = K.placeholder(shape=(None, 10))

    model = model_mnist(type=model_type)

    # Train an MNIST model
    tf_train(x, y, model, X_train, Y_train, data_gen, num_of_epochs=args.epochs)

    # Finally print the result!
    test_error = tf_test_error_rate(model, x, X_test, Y_test)
    print('Test error: %.1f%%' % test_error)
    save_model(model, model_name)
    json_string = model.to_json()
    with open(model_name+'.json', 'w') as f:
        f.write(json_string)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to model")
    parser.add_argument("--type", type=int, help="model type", default=1)
    parser.add_argument("--epochs", type=int, default=6, help="number of epochs")
    args = parser.parse_args()

    main(args.model, args.type)
