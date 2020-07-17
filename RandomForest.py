import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():
    # x is image, and y is the label
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

    x_test_origin = x_test.copy()
    # transfer image data to a 2 dimensional vector
    x_train = x_train.reshape([-1, 28 * 28])
    x_test = x_test.reshape([-1, 28 * 28])

    # train random forest
    clf = RandomForestClassifier(n_estimators=128)
    clf.fit(X=x_train[:1000], y=y_train[:1000])

    # estimate prediction
    predictions = clf.predict(x_test[:1000])
    print(accuracy_score(predictions, y_test[:1000]))
    print("Number of original training examples:", len(x_train))
    print("Number of original test examples:", len(x_test))

    plt.imshow(x_test_origin[0])
    plt.show()


if __name__ == '__main__':
    main()
