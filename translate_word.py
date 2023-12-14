import os
from tester import *
from autocorrect import *
import tensorflow as tf

def import_word_testing_data(word_folder, pmodel, wDict):
    
    blob_accs = 0
    sym_accs = 0
    base_accs = 0
    total = 0
    
    
    
    words = os.listdir(word_folder)
    if ".DS_Store" in words: words.remove(".DS_Store")
    
    for word in words:
        word_dir = os.path.join(word_folder, word)
        trials = os.listdir(word_dir)
        if ".DS_Store" in trials: trials.remove(".DS_Store")
        print("Predicting: " + word)
        correct = 0
        uncorrect = 0
        blobCorrect = 0
        
        for trial in trials:
            guess = ""
            tpath = os.path.join(word_dir, trial)
            letters = os.listdir(tpath)
            if ".DS_Store" in letters: letters.remove(".DS_Store")
            letters.sort(key=(lambda e: int(e[1])))
            for letter in letters:
                full_path_letter = os.path.join(tpath, letter)
                guess += predict(full_path_letter, pmodel)
            uncorrected = guess.lower()
            corrected = symCheckWord(guess.lower(), wDict)
            blob = blobCheckWord(guess.lower())
            
            # print(corrected)
            if uncorrected == word:
                uncorrect += 1
            if corrected == word:
                correct += 1
            if blob == word:
                blobCorrect += 1
        
        print("Sym corrected accuracy for " + word + ":", correct / 100)
        print("Blob corrected accuracy for " + word + ":", blobCorrect / 100)
        print("Uncorrected accuracy for " + word + ":", uncorrect / 100)
        
        blob_accs += blobCorrect
        sym_accs += correct
        base_accs += uncorrect
        total += 100
    return blob_accs / total, sym_accs / total, base_accs / total
        
        
                
            

def main():
    xtrain, ytrain = import_data("asl_alphabet_train_skin", 0.0, 0.8)
    model = fit(xtrain, ytrain, n_epochs = 10, prob=True, verbosity=0)
    # prob_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    # wordDict = fileToDict("freq.csv")
    spellCheck = SymSpell()
    spellCheck.load_dictionary("freq.txt", 0, 1, " ")
    b, s, n = import_word_testing_data("words_skin", model, spellCheck)
    print("Blob Acc: ", b)
    print("Sym Acc: ", s)
    print("Base Acc: ", n)
    

if __name__ == "__main__":
    main()