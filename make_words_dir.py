import random
import os
import shutil

REPS = 100

def read_file(filename): 
    f = open(filename, "r")
    word_list = f.read().split("\n")
    f.close()
    return word_list

def main():
    word_dict = read_file("dict.txt")
    common_dict = read_file("common.txt")

    l = len(common_dict)
    
    for word in common_dict:
        try:
            os.mkdir(os.path.join("words_skin", word), mode = 0o777)
        except:
            pass
        
        for i in range(REPS):
            try:
                os.mkdir(os.path.join("words_skin", word, str(i)), mode = 0o777)
            except:
                pass
            
            ctr = 0
            for letter in word:
                potential = os.listdir(os.path.join("asl_alphabet_train_skin", letter))
                shutil.copy(os.path.join("asl_alphabet_train_skin", letter, potential[random.randint(0, len(potential) - 1)]), os.path.join("words_skin", word, str(i), letter + str(ctr) + ".png"))
                ctr += 1

        
    # for word in common_dict:
    #     for letter in word:
    #         print(letter, end=" ")
    #     print()
        


if __name__ == "__main__":
    main()