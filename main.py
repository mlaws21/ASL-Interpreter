import sys 
import numpy as np 
import tensorflow as tf
from helper import *
from translate_word import predict_word
from validator import run_validator
from make_words_dir import make_word_file
from skin_image_and_save import skin_file
from train_valid_test_Splitter import write_data
from video_interpreter import video


#model_name = "final" "standard" "baseline"
def driver(model_name="final", preproc=True, ablation_on=False, test_on=False, adversarial_on=False, valid_on=False, display_model=False, prob_on=False):
    
    # model_name = "final" # "standard" "baseline"
    
    imported_evaluation_sets = []
    
    xtrain = None
    ytrain = None
    # xtrain, ytrain, xvalid, yvalid, xtest, ytest, xadversarial, yadversarial, xablation, yablation
    
    if preproc:
        #train
        xtrain, ytrain = import_data("datasets/train/")
        xsynth, ysynth = import_data("datasets/synth_train/")
        xtrain.extend(xsynth)
        ytrain.extend(ysynth)
        xtrain = np.array(xtrain)
        ytrain = np.array(ytrain)
    
        if adversarial_on:
            xadversarial, yadversarial = import_data("datasets/adversarial/")
            xadversarial = np.array(xadversarial)
            yadversarial = np.array(yadversarial)
            imported_evaluation_sets.append((xadversarial, yadversarial, "Adversarial"))
            
        if ablation_on:
            # this dataset is long so we only take the first 500-600 images   
            xablation, yablation = import_data("datasets/ablation/", 0, 0.1)
            xablation = np.array(xablation)
            yablation = np.array(yablation)
            imported_evaluation_sets.append((xablation, yablation, "Ablation"))
            
            
        if valid_on:
            xvalid, yvalid = import_data("datasets/valid/")
            xvalid = np.array(xvalid)
            yvalid = np.array(yvalid)
            imported_evaluation_sets.append((xvalid, yvalid, "Validation"))
            
            
        if test_on:
            xtest, ytest = import_data("datasets/test/")
            xtest = np.array(xtest)
            ytest = np.array(ytest)
            imported_evaluation_sets.append((xtest, ytest, "Test"))
        
    else:
        #train
        nop_xtrain, nop_ytrain = import_data("nop_datasets/base/", start=0, end=0.8, rescale=True)
        nop_xsynth, nop_ysynth = import_data("nop_datasets/synth_train/", rescale=True)
        nop_xtrain.extend(nop_xsynth)
        nop_ytrain.extend(nop_ysynth)
        xtrain = np.array(nop_xtrain)
        ytrain = np.array(nop_ytrain)
        
        
        if adversarial_on:
            nop_xadversarial, nop_yadversarial = import_data("nop_datasets/adversarial/", rescale=True)
            xadversarial = np.array(nop_xadversarial)
            yadversarial = np.array(nop_yadversarial)
            imported_evaluation_sets.append((xadversarial, yadversarial, "Adversarial"))
            
            
        if ablation_on:
            nop_xablation, nop_yablation = import_data("nop_datasets/ablation/", 0, 0.1, rescale=True)
            xablation = np.array(nop_xablation)
            yablation = np.array(nop_yablation)
            imported_evaluation_sets.append((xablation, yablation, "Ablation"))
            
            
        if valid_on:
            nop_xvalid, nop_yvalid = import_data("nop_datasets/base/", start=0.8, end=0.9, rescale=True)
            xvalid = np.array(nop_xvalid)
            yvalid = np.array(nop_yvalid)
            imported_evaluation_sets.append((xvalid, yvalid, "Validation"))
            
        
        if test_on:   
            nop_xtest, nop_ytest = import_data("nop_datasets/base/", start=0.9, end=1, rescale=True)
            xtest = np.array(nop_xtest)
            ytest = np.array(nop_ytest)
            imported_evaluation_sets.append((xtest, ytest, "Test"))
            
        
    
    
    baseline = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(70, 70, 3)),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(29)

    ])
    
    standard_nn = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(RESIZE, RESIZE, 3)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(29)

    ])
    
    final = tf.keras.Sequential([ #best model
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
    
    which_model = {"final": final, "standard": standard_nn, "baseline": baseline}
    if model_name not in which_model:
        return -1
    
    if display_model: 
        which_model[model_name].summary()
    
    learned_model = fit_model(which_model[model_name], xtrain, ytrain, n_epochs=10, prob=prob_on)
    
    for i in imported_evaluation_sets:
        print(i[2] + " Accuracy:", eval_model(learned_model, i[0], i[1]))
    return learned_model


def main():
    # run evaluations of different models with this code
    # model choices are final, standard, and baseline
    # set the flags to specify evaluation metrics
    if len(sys.argv) == 1:
        driver(valid_on=True, test_on=True, ablation_on=True, adversarial_on=True, preproc=True)
        return 0
        

    if sys.argv[1] == "word":
    # predict a word with this code:
        model = driver(prob_on=True)
        print(predict_word(model, sys.argv[2]))
        return 0
        
    
    # run the validation search with this code:
    if sys.argv[1] == "validator":
        run_validator()
        return 0
        
    
    # preprocess an set of images with this code:
    if sys.argv[1] == "preproc":
        skin_file(sys.argv[2], sys.argv[3])
        return 0
        
        
    # skin_file(<data>, <new folder for skinned data>)
    
    # split data into subsets with this code:
    #write_data(<in folder>, <out folder>, <start proportion>, <end proportion>)
    
    # make a word file for a new word with this code: 
    #make_word_file(<word>, <folder to save images to>, <folder to draw images from>, <number repititions>)
    
    # run the video interpreter with this code:
    if sys.argv[1] == "video":
        model = driver(prob_on=True)
        video(model)
        return 0
    return -1
    
if __name__ == "__main__":
    main()