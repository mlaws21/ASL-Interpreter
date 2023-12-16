import cv2 
import numpy as np 
import os 
import tensorflow as tf
import random

RESIZE = 70
# TODO try a nn layer to resize

lab = {
    "A" : 0,
    "B" : 1,
    "C" : 2,
    "D" : 3,
    "E" : 4,
    "F" : 5,
    "G" : 6,
    "H" : 7,
    "I" : 8,
    "J" : 9,
    "K" : 10,
    "L" : 11,
    "M" : 12,
    "N" : 13,
    "O" : 14,
    "P" : 15,
    "Q" : 16,
    "R" : 17,
    "S" : 18,
    "T" : 19,
    "U" : 20,
    "V" : 21,
    "W" : 22,
    "X" : 23,
    "Y" : 24,
    "Z" : 25,
    "none" : 26,
    "del" : 27,
    "space" : 28,

}

def import_data(folder, start=0, end=1, rescale=False):
    xtrain = []
    ytrain = []

    # lab = 0
    letters = sorted(os.listdir(folder))
    if ".DS_Store" in letters: letters.remove(".DS_Store")
    # if letters.count("nothing") > 0: letters.remove("nothing")

    for letter in letters:
        images = os.listdir(os.path.join(folder, letter))
        if ".DS_Store" in images: images.remove(".DS_Store")

        random.Random(2).shuffle(images)
        for i in images[int(start*len(images)) : int(end*len(images))]:
            
            img = cv2.imread(os.path.join(folder, letter, i))
            if rescale:
                xtrain.append(cv2.resize(img, (70, 70)))
            else:
                xtrain.append(img)
            ytrain.append(lab[letter])

        # lab += 1
        print(letter, end=" ", flush=True)

    print()
    # return np.array(xtrain), np.array(ytrain)
    return xtrain, ytrain

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

def fit_model(m, xtrain, ytrain, prob=False, n_epochs=10):
    

    m.compile(optimizer='adam', # lr 0.001
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='accuracy',
            min_delta=0.001,
            patience=0,
            verbose=1,
        )
    ]
    m.fit(xtrain, ytrain, epochs=n_epochs, callbacks=my_callbacks)
    
    if prob:
        return tf.keras.Sequential([m, tf.keras.layers.Softmax()])
    else:  
        return m
    
def fit_best_model(xtrain, ytrain, prob=False, n_epochs=10):
    m = tf.keras.Sequential([ #best model
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(RESIZE, RESIZE, 3)), # strides default to 1
        tf.keras.layers.MaxPooling2D((2, 2)), #strides defaulting to pool_size = (2, 2)
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        
        tf.keras.layers.Dense(29)
    ])

    m.compile(optimizer='adam', # lr 0.001
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='accuracy',
            min_delta=0.001,
            patience=0,
            verbose=1,
        )
    ]
    m.fit(xtrain, ytrain, epochs=n_epochs, callbacks=my_callbacks)
    
    if prob:
        return tf.keras.Sequential([m, tf.keras.layers.Softmax()])
    else:  
        return m


def eval_model(m, xeval, yeval):
    
    test_loss, eval_acc = m.evaluate(xeval, yeval, verbose=2)

    return eval_acc
    

def main():

    xtrain, ytrain = import_data("datasets/train/", 0, 1)
    xstrain, ystrain = import_data("datasets/synth_train/", 0, 1)
    xtrain.extend(xstrain)
    ytrain.extend(ystrain)
    xtest, ytest = import_data("datasets/test/", 0, 1)
    xcross, ycross = import_data("datasets/cross/", 0, 1)
    
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)
    xtest = np.array(xtest)
    ytest = np.array(ytest)
    xcross = np.array(xcross)
    ycross = np.array(ycross)


    

if __name__ == "__main__":
    main()