
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, callbacks


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


df_ravdess = pd.read_csv("training_data_ravdess.csv", index_col=0)
df_crema = pd.read_csv("training_data_crema.csv", index_col=0)

df = df_ravdess.append(df_crema)

print(df.head())

# All rows, all cols but the last one (which is the labels)
X = df.iloc[:, :-1].values
Y = df["label"].values

encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()


# splitting data
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# print(x_train)
# making our data compatible to model.
x_train = np.expand_dims(x_train, axis=2)
# print(x_train)
x_test = np.expand_dims(x_test, axis=2)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


model = keras.models.Sequential(
    [
        layers.BatchNormalization(),
        layers.Conv1D(
            256,
            kernel_size=5,
            strides=1,
            padding="same",
            activation="relu",
            input_shape=(x_train.shape[1], 1),
        ),
        layers.MaxPooling1D(pool_size=5, strides=2, padding="same"),
        layers.Conv1D(256, kernel_size=5, strides=1, padding="same", activation="relu"),
        layers.MaxPooling1D(pool_size=5, strides=2, padding="same"),
        layers.Conv1D(128, kernel_size=5, strides=1, padding="same", activation="relu"),
        layers.MaxPooling1D(pool_size=5, strides=2, padding="same"),
        layers.Dropout(0.2),
        layers.Conv1D(64, kernel_size=5, strides=1, padding="same", activation="relu"),
        layers.MaxPooling1D(pool_size=5, strides=2, padding="same"),
        layers.Flatten(),
        layers.Dense(units=32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(units=8, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


rlrp = callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.4, verbose=0, patience=2, min_lr=0.0000001
)

history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=50,
    validation_data=(x_test, y_test),
    callbacks=[rlrp],
)

print(
    "Accuracy of our model on test data : ",
    model.evaluate(x_test, y_test)[1] * 100,
    "%",
)

model.save("model.h5")

model.summary()
