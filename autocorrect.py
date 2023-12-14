# knn; k = 1 autocorrector
# distance = number of letters off * λ / frequency

# can build in sign similarity into the distance model
import csv
import numpy as np
import statistics as stat
import math
from symspellpy import SymSpell
from textblob import TextBlob


		# Driver code


# This code is contributed by divyesh072019.


def fileToDict(filename, exitEarly):
    tempDict = {}
    f = open(filename, "r") 
    csvf = csv.reader(f)
    c = []
    ctr = 0
    for line in csvf:
        if len(line[0]) < 15:
            tempDict[(line[0])] = line[1]
            c.append(int(line[1]))
            ctr += 1
            if (ctr > exitEarly):
                break

            
            
    mean = stat.mean(c)
    stdev = stat.stdev(c)
    for i in tempDict:
        tempDict[i] = 1 / (1 + math.e ** (-((int(tempDict[i]) - mean) / stdev)))
    f.close()
    
    return tempDict
        
    
def symCheckWord(word, checkDict):
    corrected = checkDict.lookup_compound(word, 2)
    return(corrected[0].term)
    
def blobCheckWord(word):
    return TextBlob(word).correct()

def main():
    wordDict = fileToDict("freq.csv", 10000)

    myString = "In a qiuet village nsetled between rolling hills, a mystreious book apepared overnigth in teh town squaer. Its lteaher cover whispered of unotld secrets. The townfsolk, curious yte wary, gatherde aroudn as teh book opened on ist onw. Wodrs shimmered on teh pgeas, weaivng tlaes of forgottne draems adn ltos loves. Each reader saw a perosnalized story, a refletcion of theri deepest desires adn fears. Teh book beacme a cherished patr of the commniuty, ist maigc brniging solace adn inspiration. As seaosns changed, teh vilaleg thirved, forever tocuhed by teh enchatnign words that emerged form the mysteroius tome in the suqare."

    spellCheck = SymSpell()
    spellCheck.load_dictionary("freq.txt", 0, 1, " ")
    print(symCheckWord(myString, spellCheck))
    print(blobCheckWord(myString))
    



if __name__ == "__main__":
    main()