import cv2
import mediapipe as mp
import numpy as np
from helper import *

mpHands = mp.solutions.hands
hands = mpHands.Hands()
# mpDraw = mp.solutions.drawing_utils

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

def skin(image, edit):
    
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

        # checking whether a hand is detected
    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0] # working with each hand

        h, w, c = image.shape
        for i in pairs:
            if i[2] == "h":
                continue
            cx, cy = int(handLms.landmark[i[0]].x * w), int(handLms.landmark[i[0]].y * h)
            cxp, cyp = int(handLms.landmark[i[1]].x * w), int(handLms.landmark[i[1]].y * h)
            cv2.line(edit, (cx, cy), (cxp, cyp), key_to_color[i[2]], 4)
        
        for i in tips: 
            cx, cy = int(handLms.landmark[i].x * w), int(handLms.landmark[i].y * h)
            cv2.circle(edit, (cx, cy), 8, key_to_color[i], cv2.FILLED)
    
        return edit
    
    else: return None

def video(fit_model):
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    crop = None
    # adj = 0
    while True:
        success, image = cap.read()
        relay = image.copy()
        imageRGB = cv2.cvtColor(relay, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)
        h, w, c = relay.shape
        # blank = np.zeros((h,w,3), np.uint8)
        
            # checking whether a hand is detected
        if results.multi_hand_landmarks:
            handLms = results.multi_hand_landmarks[0] # working with each hand

            # want to make a bounding box
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h

            # print(int(handLms.landmark[8].x * w))
            # print(int(handLms.landmark[8].y * h))
            
            for lm in handLms.landmark:
                if int(lm.x * w) > x_max:
                    x_max = int(lm.x * w)

                if int(lm.x * w) < x_min:
                    x_min = int(lm.x * w)
                    
                if int(lm.y * h) > y_max:
                    y_max = int(lm.y * h)
                    
                if int(lm.y * h) < y_min:
                    y_min = int(lm.y * h)

            adj = ((max((x_max - x_min), (y_max - y_min))) // 2) + 100
            x_mid = ((x_max - x_min) // 2) + x_min
            y_mid = ((y_max - y_min) // 2) + y_min
            
            
            # cv2.rectangle(image, (x_mid - adj, y_mid - adj), (x_mid + adj, y_mid + adj), (255, 255, 255), 10 )
            
            x1, y1, x2, y2 = (x_mid - adj), (y_mid - adj), (x_mid + adj), (y_mid + adj)
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 10 )
            
            if not (x1 < 0 or x2 > w or y1 < 0 or y2 > h):

                crop = image[x1:x2, y1:y2]
                crop = cv2.resize(crop, (200, 200))
                # h, w, c = crop.shape
                blank = np.zeros((200, 200 ,3), np.uint8)
                skinned  = skin(crop, blank)
                # skinned = crop
                # skinned = cv2.resize(skinned, (70, 70))
                # skinned  = skin(crop, blank)
                if skinned is not None:
                    
                    skinned = cv2.resize(skinned, (70, 70))
                    # skinned = skinned / 255.0
                    # cv2.imshow("sign", cv2.flip(skinned, 1))
                    # cv2.imwrite("crop.png", cv2.flip(crop, 1))
                    
                    # ignore double letters for now
                    
                    print(img_predict(skinned, fit_model), end="", flush=True)

        cv2.imshow("Output", cv2.flip(relay, 1))

        
        cv2.waitKey(1)
        
def main():
    xtrain, ytrain = import_data("asl_alphabet_train_skin", 0.0, 0.8)
    model = fit_best_model(xtrain, ytrain, n_epochs=10, prob=True)
    # model = ""
    video(model)
    
        
        
if __name__ == "__main__":
    main()