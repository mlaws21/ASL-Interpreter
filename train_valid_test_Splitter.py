import cv2 
import numpy as np 
import os 
import tensorflow as tf
import random
import shutil


def write_data(folder, set, start, end):


    letters = sorted(os.listdir(folder))
    if ".DS_Store" in letters: letters.remove(".DS_Store")


    for letter in letters:
        
        try:
            os.mkdir(os.path.join("datasets", set, letter), mode = 0o777)
        except:
            pass
        
        images = os.listdir(os.path.join(folder, letter))
        if ".DS_Store" in images: images.remove(".DS_Store")

        random.Random(2).shuffle(images)
        for i in images[int(start*len(images)) : int(end*len(images))]:
            
            shutil.copy(os.path.join(folder, letter, i), os.path.join("datasets", set , letter))

            

        print(letter, end=" ", flush=True)

    print()


def main():
    write_data("asl_alphabet_train_skin", "train", 0, 0.8)
    write_data("asl_alphabet_train_skin", "valid",  0.8, 0.9)
    write_data("asl_alphabet_train_skin", "test",  0.9, 1)
    

if __name__ == "__main__":
    main()