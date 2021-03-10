
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, callbacks


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


df_ravdess = pd.read_csv("training_data_ravdess.csv", index_col=0)
df_crema = pd.read_csv("training_data_crema.csv", index_col=0)
df_savee = pd.read_csv("training_data_savee.csv", index_col=0)
df_tess = pd.read_csv("training_data_tess.csv", index_col=0)

df = df_ravdess.append(df_crema).append(df_savee).append(df_tess)

print(df.head())
# print(dir(df))

# All rows, all cols but the last one (which is the labels)
X = df.iloc[:, :-1].values
Y = df["label"].values


encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
print("labels", Y)


# splitting data
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
print("train, test split")
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(x_train)


model = keras.models.Sequential(
    [
        layers.Input(shape=(162)),
        layers.BatchNormalization(),
        layers.Dense(512, kernel_initializer="he_normal", activation="elu"),
        layers.Dense(512, kernel_initializer="he_normal", activation="elu"),
        layers.Dense(512, kernel_initializer="he_normal", activation="elu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(256, kernel_initializer="he_normal", activation="elu"),
        layers.Dense(256, kernel_initializer="he_normal", activation="elu"),
        layers.Dense(256, kernel_initializer="he_normal", activation="elu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(128, kernel_initializer="he_normal", activation="elu"),
        layers.Dropout(0.2),
        layers.Dense(64, kernel_initializer="he_normal", activation="elu"),
        layers.Dropout(0.2),
        layers.Dense(units=8, activation="softmax"),
    ]
)

model.compile(optimizer="nadam", loss="categorical_crossentropy", metrics=["accuracy"])


rlrp = callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.4, verbose=0, patience=2, min_lr=0.0000001
)

history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=[rlrp],
)

print(
    "Accuracy of our model on test data : ",
    model.evaluate(x_test, y_test)[1] * 100,
    "%",
)

model.save("model2.h5")

model.summary()
