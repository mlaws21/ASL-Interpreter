import os
from helper import *
from autocorrect import *
import tensorflow as tf

def predict_word_testing_data(word_folder, pmodel, wDict, outfile):
    
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
            
            # print(corrected)
            if uncorrected == word:
                uncorrect += 1
            if corrected == word:
                correct += 1
        
        lines = [("Sym corrected accuracy for " + word + ": " + str(correct / 50) + "\n"),
                 ("Uncorrected accuracy for " + word + ": " + str(uncorrect / 50) + "\n")
                ]
        outfile.writelines(lines)
        
        sym_accs += correct
        base_accs += uncorrect
        total += 50
    return sym_accs / total, base_accs / total
        
# this code predicts a single word  
# should pass it a probability model
#input should be a folder with an image of each letter in the form <letter><0 indexed position in word>.png
# for example hi: h0.png, i1.png
def predict_word(pmodel, word_folder):
    wDict = SymSpell()
    guess = ""
    letters = os.listdir(word_folder)
    if ".DS_Store" in letters: letters.remove(".DS_Store")
    letters.sort(key=(lambda e: int(e[1])))
    for letter in letters:
        letter_path = os.path.join(word_folder, letter)
        guess += predict(letter_path, pmodel)
    return symCheckWord(guess.lower(), wDict)
    
    
                
            
# this code runs word results check
def main():
    
    xtrain, ytrain = import_data("datasets/train/", 0, 1)
    xstrain, ystrain = import_data("datasets/synth_train/", 0, 1)
    xtrain.extend(xstrain)
    ytrain.extend(ystrain)
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)
    model = fit_best_model(xtrain, ytrain, prob=True, n_epochs=1)
    print(predict_word(model, "word_test/above/0/"))
    # spellCheck = SymSpell() # default max edit dist = 2
    # spellCheck.load_dictionary("freq.txt", 0, 1, " ")
    # f = open("TranslateResults.txt", "a")
    # s, n = predict_word_testing_data("word_validation", model, spellCheck, f)
    # f.close()
    # print("Sym Acc: ", s)
    # print("Base Acc: ", n)
    

if __name__ == "__main__":
    main()