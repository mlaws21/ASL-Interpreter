import cv2 
import numpy as np 
import os 
import cv2
import tensorflow as tf
import random

RESIZE = 70
# TODO try a nn layer to resize


def import_data(folder, start, end):
    xtrain = []
    ytrain = []

    lab = 0
    letters = sorted(os.listdir(folder))
    if ".DS_Store" in letters: letters.remove(".DS_Store")
    # if letters.count("nothing") > 0: letters.remove("nothing")

    for letter in letters:
        images = os.listdir(os.path.join(folder, letter))
        if ".DS_Store" in images: images.remove(".DS_Store")

        random.Random(2).shuffle(images)
        for i in images[int(start*len(images)) : int(end*len(images))]:
            
            img = cv2.imread(os.path.join(folder, letter, i))

            xtrain.append(img)
            ytrain.append(lab)

        lab += 1
        print(letter, end=" ", flush=True)

    print()
    return np.array(xtrain), np.array(ytrain)

def fit(xtrain, ytrain, n_epochs=10, prob=False, verbosity=1):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(RESIZE, RESIZE, 3)), # strides default to 1
        tf.keras.layers.MaxPooling2D((2, 2)), #strides defaulting to pool_size = (2, 2)
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        
        tf.keras.layers.Dense(29)

    ])

    model.compile(optimizer='adam', # lr 0.001
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='accuracy',
            min_delta=0.001,
            patience=0,
            verbose=verbosity,
        )
    ]
    model.fit(xtrain, ytrain, epochs=n_epochs, callbacks=my_callbacks)
    
    if prob:
        return tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    else:  
        return model

def display_prediction(raw):
    conv = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "none", "space"]

    ind = max(raw)
    print("Prediction: ", conv[list(raw).index(ind)])
    print("Distribution: ")
    for i in range(len(raw)):
        print(conv[i] + ": ", round(raw[i], 2))
    
def predict(img_filename, model):
    conv = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "none", "space"]
    openImg = cv2.imread(img_filename)
    raw = model.predict(np.array([openImg]), verbose=0)[0]
    ind = max(raw)
    pred = conv[list(raw).index(ind)]
    if pred == "none":
        return ""
    elif pred == "space":
        return " "
    elif pred == "del":
        return "<"
    else:
        return pred
    # display_prediction(probability_model.predict(np.array([openImg]))[0])

def img_predict(img, model):
    conv = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "none", "space"]

    raw = model.predict(np.array([img]), verbose=0)[0]
    ind = max(raw)
    pred = conv[list(raw).index(ind)]
    if pred == "none":
        return ""
    elif pred == "space":
        return " "
    elif pred == "del":
        return "<"
    else:
        return pred

def main():
    xntrain, yntrain = import_data("datasets/train/", 0, 1)
    xstrain, ystrain = import_data("datasets/synth_train/", 0, 1)
    xtrain = np.concatenate(xntrain, xstrain, axis=0)
    ytrain = np.concatenate(yntrain, ystrain, axis=0)
    xvalid, yvalid = import_data("datasets/valid/", 0, 1)
    xcross, ycross = import_data("datasets/cross/", 0, 1)
    
    model = fit(xtrain, ytrain)

    test_loss, test_acc = model.evaluate(xvalid, yvalid, verbose=2)

    print('\nValidation accuracy:', test_acc)

    test_loss, cross_acc = model.evaluate(xcross, ycross, verbose=2)

    print('\nCross accuracy:', cross_acc)
    
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    
    
    img = ""
    while img != "q" or img != "quit":
        img = input("Enter Path to Image to Predict: ")
        print(predict(img, probability_model))



if __name__ == "__main__":
    main()