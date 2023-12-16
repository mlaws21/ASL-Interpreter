import random
import os
import shutil



def read_file(filename): 
    f = open(filename, "r")
    word_list = f.read().split("\n")
    f.close()
    return word_list

def make_word_file(word, wordFolder, candidate_folder, reps=50):
    try:
        os.mkdir(os.path.join(wordFolder, word), mode = 0o777)
    except:
        pass
    
    for i in range(reps):
        try:
            os.mkdir(os.path.join(wordFolder, word, str(i)), mode = 0o777)
        except:
            pass
        
        ctr = 0
        for letter in word:
            potential = os.listdir(os.path.join(candidate_folder, letter))
            shutil.copy(os.path.join(candidate_folder, letter, potential[random.randint(0, len(potential) - 1)]), os.path.join(wordFolder, word, str(i), letter + str(ctr) + ".png"))
            ctr += 1

def main():
    # word_dict = read_file("dict.txt")
    common_dict = read_file("common.txt")

    # l = len(common_dict)
    
    for word in common_dict:
        make_word_file(word, "word_test/", "datasets/test/")

        
    # for word in common_dict:
    #     for letter in word:
    #         print(letter, end=" ")
    #     print()
        


if __name__ == "__main__":
    main()