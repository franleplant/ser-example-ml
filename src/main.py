from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
print(keras.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# for i in range(25):
#     pyplot.subplot(5, 5, i + 1)
#     pyplot.imshow(X_train_full[i + 100], cmap=pyplot.get_cmap("gray"))
#
# pyplot.show()

X_train = X_train_full[5000:] / 255.0
X_valid = X_train_full[:5000] / 255.0
y_train = y_train_full[5000:]
y_valid = y_train_full[:5000]

X_test = X_test / 255.0


model = keras.models.Sequential(
    [
        layers.Flatten(input_shape=[28, 28]),
        layers.Dense(300, activation="relu"),
        layers.Dense(100, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

model.summary()

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

pred = model.predict_classes(X_test[:25])

print(pred)

class_names = [
    "t-shirt/top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
]

for p in pred:
    print(class_names[p])

for i in range(25):
    pyplot.subplot(5, 5, i + 1)
    pyplot.imshow(X_test[i], cmap=pyplot.get_cmap("gray"))

pyplot.show()
