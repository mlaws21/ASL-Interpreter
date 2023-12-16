from helper import *


def fit_series(potential_models, xtrain, ytrain, xvalid, yvalid, xcross, ycross, f, n_epochs=10):
    out_models = []
    ctr = 0
    for i in potential_models:

        i.compile(optimizer='adam', # lr 0.001
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        
        my_callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='accuracy',
                min_delta=0.001,
                patience=0,
                verbose=1,
            )
        ]
        i.fit(xtrain, ytrain, epochs=n_epochs, callbacks=my_callbacks)
        out_models.append(i)
        print("Model", ctr)
        test_loss, test_acc = i.evaluate(xvalid, yvalid, verbose=0)

        print('\nValidation accuracy:', test_acc)

        test_loss, cross_acc = i.evaluate(xcross, ycross, verbose=0)
        print('\nCross accuracy:', cross_acc)
        f.write("Model " + str(ctr) + '\nValidation accuracy:' + str(test_acc) + '\nCross accuracy:' + str(cross_acc))
        
        ctr += 1
    return out_models
    
    
def run_validator():
    m0 = tf.keras.Sequential([ # model 0
        tf.keras.layers.Conv2D(32, (10, 10), activation='relu', input_shape=(RESIZE, RESIZE, 3)), # strides default to 1
        tf.keras.layers.MaxPooling2D((3, 3)), #strides defaulting to pool_size = (2, 2)
        tf.keras.layers.Conv2D(64, (4, 4), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        
        tf.keras.layers.Dense(29)

    ])
    
    m1 = tf.keras.Sequential([ # model 1
        tf.keras.layers.Conv2D(32, (4, 4), activation='relu', input_shape=(RESIZE, RESIZE, 3)), # strides default to 1
        tf.keras.layers.MaxPooling2D((2, 2)), #strides defaulting to pool_size = (2, 2)
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        
        tf.keras.layers.Dense(29)

    ])
    
    m2 = tf.keras.Sequential([ # model 2 
        tf.keras.layers.Conv2D(32, (4, 4), activation='relu', input_shape=(RESIZE, RESIZE, 3)), # strides default to 1
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        
        tf.keras.layers.Dense(29)

    ])
    
    m3 = tf.keras.Sequential([ # model 3
        tf.keras.layers.Conv2D(32, (4, 4), activation='relu', input_shape=(RESIZE, RESIZE, 3)), # strides default to 1
        tf.keras.layers.MaxPooling2D((2, 2)), #strides defaulting to pool_size = (2, 2)
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        
        tf.keras.layers.Dense(29)

    ])
    
    m4 = tf.keras.Sequential([ # model 4 
        tf.keras.layers.Conv2D(32, (4, 4), activation='relu', input_shape=(RESIZE, RESIZE, 3)), # strides default to 1
        tf.keras.layers.MaxPooling2D((2, 2)), #strides defaulting to pool_size = (2, 2)
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        
        tf.keras.layers.Dense(29)

    ])
    
    m5 = tf.keras.Sequential([ #model 5 
        tf.keras.layers.Conv2D(32, (10, 10), activation='relu', input_shape=(RESIZE, RESIZE, 3)), # strides default to 1
        tf.keras.layers.MaxPooling2D((3, 3)), #strides defaulting to pool_size = (2, 2)
        tf.keras.layers.Conv2D(64, (4, 4), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        
        tf.keras.layers.Dense(29)

    ])
    
    m6 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (10, 10), activation='relu', input_shape=(RESIZE, RESIZE, 3)), # strides default to 1
        tf.keras.layers.MaxPooling2D((3, 3)), #strides defaulting to pool_size = (2, 2)
        tf.keras.layers.Conv2D(64, (4, 4), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        
        tf.keras.layers.Dense(29)

    ])
    
    m7 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (4, 4), activation='relu', input_shape=(RESIZE, RESIZE, 3)), # strides default to 1
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        
        tf.keras.layers.Dense(29)

    ])
    
    m8 = tf.keras.Sequential([
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
    
    m9 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(RESIZE, RESIZE, 3)), # strides default to 1
        tf.keras.layers.MaxPooling2D((2, 2)), #strides defaulting to pool_size = (2, 2)
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        
        tf.keras.layers.Dense(29)
    ])
    
    m10 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (8, 8), activation='relu', input_shape=(RESIZE, RESIZE, 3)), # strides default to 1
        tf.keras.layers.MaxPooling2D((2, 2)), #strides defaulting to pool_size = (2, 2)
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        
        tf.keras.layers.Dense(29)
    ])
    
    m11 = tf.keras.Sequential([
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
    
    m12 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(RESIZE, RESIZE, 3)), # strides default to 1
        tf.keras.layers.MaxPooling2D((2, 2)), #strides defaulting to pool_size = (2, 2)
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        
        tf.keras.layers.Dense(29)
    ])
    
    
    xtrain, ytrain = import_data("datasets/train/", 0, 1)
    xstrain, ystrain = import_data("datasets/synth_train/", 0, 1)
    xtrain.extend(xstrain)
    ytrain.extend(ystrain)
    xvalid, yvalid = import_data("datasets/valid/", 0, 1)
    xcross, ycross = import_data("datasets/cross/", 0, 1)
    
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)
    xvalid = np.array(xvalid)
    yvalid = np.array(yvalid)
    xcross = np.array(xcross)
    ycross = np.array(ycross)
    
    models = [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12]
    
    file = open("validResults.txt", "a+")
    
    fit_models = fit_series(models, xtrain, ytrain, xvalid, yvalid, xcross, ycross, file)

    file.close()
    
    ctr = 1
    for i in fit_models:
        print("Model", ctr)
        test_loss, test_acc = i.evaluate(xvalid, yvalid, verbose=2)

        print('\nValidation accuracy:', test_acc)

        test_loss, cross_acc = i.evaluate(xcross, ycross, verbose=2)

        print('\nCross accuracy:', cross_acc)
    
        ctr += 1

def main():
    run_validator()

if __name__ == "__main__":
    main()