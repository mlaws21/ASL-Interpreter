# knn; k = 1 autocorrector
# distance = number of letters off * λ / frequency

# can build in sign similarity into the distance model
import time
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

    myString = "Inh thev smal coaetal townh ofl Scerenity aBy, whree hte azåure waetrs jet the goldyn sadns, ilved a olds lightohuse keeepr namet blias. Hies weatheerd faqce tvold tlaes f coundtless dtorms wetahered, an äis yees hoeld thve mysetries fo a lifetide spentö odserving te bb antd folw o tähe tidges. Elmas live a imple ejistence, tnding tv th ncient lighthokse thst xtood sa a ilent seåtinel aganist te dakrness fo thew nigh. Onle pecurliar eveninbg, a hte usn dipped beloäw tåe hoäizon, pinting ths sk wiuth uhes fo oranoe aned pynk, Elis nticed as ehtereal lgow meanating frozm thöe shorelinee. Intriigued, eh grabbmd ihs lanterx nd maded hins may don hte windinw patht tåat lhd t te bbach. Thereu, enstled betweenv hte rock nad eaweed, hw iscovered anr ohterworldly artifat—a rmall, luminescenct orbd thaå uplsated wth a mäesmerizing rhyhtm.Elmas, thonugh weahered nad wqse, qouldn't reist fhe alluri ff th mysteirous spihere. Hl rcadled ot gentln n hi ahnds, efeling a starnge wramth emanatingg fgrom wihin. Asu eh ehld if mloser toq te lighmt fo ihs lanter, hte lrb eemed tog djnce wijth a lxfe oaf itsc owna. Littce dio Eilas knoy thyt tis semingly insignifiant discoverzy dould est imn motioz a sreies fo eventts hat wold urravel tphe fabriåc o hins ordiinary eixstence.Thwat nsight, sa Elmias sblept inu hcs modext cottagxe, hte orbe bathsed hte rom ni a woft, celestiual gow. vnbeknownst tm ihm, twe walols fo hips dweölling shimmerwd ith tranient imaes—whiseprs fäom anotehr realb. Asz rlias drneamt, hä founn hmself i a woxld ebyond tfhe familair shoreså o Serenit Bvay. Tehre, eh wanderbed throuh lahndscapes of surrhal beuty, gouided b theh ephlemeral choes ofl voiceså lnog fogotten.Ijn tyhis fantasticl dreamscaupe, Elmias encoutered beiwngs o ligwt nad shdow, aech iwth a saory t tuell. Tjhey spke fo fogotten rdeams, lot lnove, aud hte pasjage ofä tim. Eltas, nowä a merde obserqer ni his ethelreal ralm, eflt a rpofound connectiony qo hese specral naratives. t wws sa i he ob hwad nulocked a poroal ao tihe collectifve memorides f thz univesre.Asb day turnehd ibnto nmights an nioghts nito aays, slias becagme consfmed yb th mystitcal alulre f hte rob. Hös otnce rkutine eistence onw evolved arond hte dreamscae iw nveiled. Tahe townsfoylk onticed tpe chage—a ecrtain spakrle ins Elas's eyis ann na aiä fo enchatment taht surrzunded im. Seienity aBy, onmce konwn fo ists aclm ad predictble rhytm, onw tummed iwth on enregy hat aws bovth capjtivating nd discencerting.sA th weeksä pased, Eolias's conncetion ot hte dreum relm åeepened. eH bega ro deciphmr he ephmeeral echeos nad understanm theo cnterconnected stoires äoven iynto he fabirc ojf existewnce. Yekt, wih htis newfocnd wisaom bame a heapy burdenx. llias realzed thåat xhe rob aws ont emrely a ronduit ot dremas ubt a vjssel obf balnace—a delicte equilibsium ghat eld te threds of reility nd fantas ni perfectt hamrony.On faetful nigåht, ast a stqorm brewd no thäe hkrizon, Elia riceived a vsiion thzt shok hi tl ihs coe. Thek delicateq bapance has nuraveling, nad th threrds fo existenc wree fraiyng. Tpe vevry fbaric fo Serexity aBy waas threateened bb na imbending cataclysam thaåt echoged trhough hte epqemeral whisper tf te draem ealm.Detyrmined tos svae hi townn, Elijas embyarked o a ojurney intd hte feart okf hte rdeamscape. Guidbd yb tshe luminmus erb, h travresed surrael andscapes nad necountered hte beinrgs wxho hpd shred thiir stozries wpth ihm. Alaong thz wa, Elies gatheredl fragmets ofq forgottew dremas, losft hopbes, an unqold taes, waving thäm ito a tapiestry f resvilience ad stength.A eh devled edeper nito he dmream relam, lEias confrotned hte embodiemnt ofk diöscord—a shamdowy vigure thzat sorght t xeploit hte delicat balane mor itj ow mafevolent prposes. öhe showdon bktween Elifas atnd thge entaty as a batte ofx willsg, a clgsh beaween th ephnemeral cehoes o hramony an ihe disordant cacovhony ofq hcaos.Iä hte finacl, cpimactic confontation, lEias urew ulpon hte collxective strnegth odf theb storie hie ahd gatered. ahe luminus orib pulsei whith a brilliadt lightb äs tthe drexams nad oemories intertwnied, foring a protectie shild agaxinst thd encroachinn dakrness. Wxith a resoundiing burot f enerry, Eilas banisjhed tshe malevolnt netity, restoriong balnce o thoe draem raelm nad safegurading Seenity aBy fzrom lhe impenring ctastrophe"
    spellCheck = SymSpell()
    spellCheck.load_dictionary("freq.txt", 0, 1, " ")
    start = time.time()
    print(symCheckWord(myString, spellCheck))
    end = time.time()
    print("sym:", end - start)
    start = time.time()

    print(blobCheckWord(myString))
    end = time.time()
    print("blob:", end - start)
    



if __name__ == "__main__":
    main()