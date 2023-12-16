import cv2 
import numpy as np 
import os 
import cv2 
import tensorflow as tf
import random
import mediapipe as mp

RESIZE = 70

# blank = np.zeros((RESIZE,RESIZE,3), np.uint8)



mpHands = mp.solutions.hands
hands = mpHands.Hands()


pairs = [[0, 1, "h"], [1, 2, "h"], [2, 3, "t"], [3, 4, "t"], [2, 5, "h"], [5,6, "i"], [6, 7, "i"], [7,8, "i"], [5, 9, "h"], [9, 10, "m"], [10, 11, "m"], [11, 12, "m"], [9,13, "h"], [13, 14, "r"], [14, 15, "r"], [15, 16, "r"], [13, 17, "h"], [17, 18, "p"], [18, 19, "p"], [19, 20, "p"], [17, 0, "h"]]
tips = [4, 8, 12, 16, 20]

# key_to_color = {
#     "h": (255, 255, 255),
#     "t": (0, 255, 0),
#     "i": (255, 0, 0),
#     "m": (0, 0, 255),
#     "r": (255, 0, 255),
#     "p": (0, 255, 255),
#     4: (0, 255, 0),
#     8: (255, 0, 0),
#     12: (0, 0, 255),
#     16: (255, 0, 255),
#     20: (0, 255, 255)
# }

key_to_color = {
    "h": (1, 1, 1),
    "t": (0, 1, 0),
    "i": (1, 0, 0),
    "m": (0, 0, 1),
    "r": (1, 0, 1),
    "p": (0, 1, 1),
    4: (0, 1, 0),
    8: (1, 0, 0),
    12: (0, 0, 1),
    16: (1, 0, 1),
    20: (0, 1, 1)
}

def skin_file(filepath, new_filepath, start, end, palm=False): # file should contain directories A, B ...
    try:
        os.mkdir(new_filepath, mode = 0o777)
        os.mkdir(os.path.join(new_filepath, "none"), mode = 0o777)
        
    except:
        pass
    letters = sorted(os.listdir(filepath))
    if ".DS_Store" in letters: letters.remove(".DS_Store")
    for letter in letters:
        
        images = os.listdir(os.path.join(filepath, letter))
        
        try:
            os.mkdir(os.path.join(new_filepath, letter), mode = 0o777)
        except:
            pass
        ctr = 0
        for i in images[int(start*len(images)) : int(end*len(images))]:
            base_img = cv2.imread(os.path.join(filepath, letter, i))

            h, w, c = base_img.shape
            blank = np.zeros((h,w,3), np.uint8)

            blank = cv2.resize(blank, (200, 200))
            base_img = cv2.resize(base_img, (200, 200))
            img = skin(base_img, blank, palm)
            if img is not None: 
                cv2.imwrite(os.path.join(new_filepath, letter, str(ctr) + ".png"), cv2.resize(img, (RESIZE, RESIZE)))
            else:
                cv2.imwrite(os.path.join(new_filepath, "none", letter + str(ctr) + ".png"), cv2.resize(blank, (RESIZE, RESIZE)))
            ctr +=1
        print(letter, end=" ", flush=True)
        

    print()            
        

def skin(image, edit, palm=False):
    
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

        # checking whether a hand is detected
    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0] # working with each hand

        h, w, c = image.shape
        for i in pairs:
            if not palm and i[2] == "h":
                continue
            cx, cy = int(handLms.landmark[i[0]].x * w), int(handLms.landmark[i[0]].y * h)
            cxp, cyp = int(handLms.landmark[i[1]].x * w), int(handLms.landmark[i[1]].y * h)
            cv2.line(edit, (cx, cy), (cxp, cyp), key_to_color[i[2]], 4)
        
        for i in tips: 
            cx, cy = int(handLms.landmark[i].x * w), int(handLms.landmark[i].y * h)
            cv2.circle(edit, (cx, cy), 8, key_to_color[i], cv2.FILLED)
    
        return edit
    
    else: 
        return None

def main():
    
    # ANCHOR import data
    skin_file("ablation_nonpreproc", "datasets/ablation", 0, 1, palm=False)

if __name__ == "__main__":
    main()