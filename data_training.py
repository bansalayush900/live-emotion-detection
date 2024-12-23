import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model
 
is_init = False
size = -1
label = []
dictionary = {}
c = 0

# Loading .npy files and creating the dataset
for i in os.listdir():
    if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
        if not(is_init):
            is_init = True 
            X = np.load(i)
            size = X.shape[0]
            y = np.array([i.split('.')[0]]*size).reshape(-1, 1)
        else:
            X = np.concatenate((X, np.load(i)))
            y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1, 1)))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c = c + 1

# Converting string labels to integers
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# Converting labels to one-hot encoding
y = to_categorical(y)

# Shuffling data
X_new = X.copy()
y_new = y.copy()
counter = 0 
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt: 
    X_new[counter] = X[i]
    y_new[counter] = y[i]
    counter = counter + 1

X = X_new
y = y_new

# Ensure X has at least 2 dimensions
if len(X.shape) == 1:
    X = X.reshape(-1, 1)

# Debugging: print the shape of X to ensure it's correct
print("X shape:", X.shape)  
print("y shape:", y.shape)

# Model definition
ip = Input(shape=(X.shape[1],))  # Correct input shape for the model
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)

# Compile the model
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train the model
model.fit(X, y, epochs=50)

# Save the model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))
