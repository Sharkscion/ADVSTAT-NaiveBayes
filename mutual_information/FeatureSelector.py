import math
import operator

from model.Word import Word

class FeatureSelector:
    def __init__(self, distinctWords):
        self.distinctWords = distinctWords

    def getRelevantWords(self):
        for k in self.distinctWords:
            self.distinctWords[k].mutualInfo = self.getMutualInfo(self.distinctWords[k])

        self.distinctWords = sorted(self.distinctWords.items(), key=lambda x:x[1].mutualInfo, reverse = True)[:200]

        # for k in self.distinctWords:
        #     print(k[0] , ": ",k[1].mutualInfo)
        self.distinctWords = {x[0]: x[1] for x in self.distinctWords}
        print("after convert back to dictionary..........................")


        return self.distinctWords


    def getMutualInfo(self, distinctWord):

        totalResults = distinctWord.presentSpamCount + distinctWord.notPresentSpamCount + \
                       distinctWord.presentLegitCount + distinctWord.notPresentLegitCount

        try:
            # P(x=0,c=spam)
            mutualInfo = (distinctWord.notPresentSpamCount/totalResults)*\
                         math.log((distinctWord.notPresentSpamCount * totalResults)/\
                                    ((distinctWord.notPresentSpamCount+distinctWord.notPresentLegitCount)*(distinctWord.notPresentSpamCount+distinctWord.presentSpamCount)),2)

        except (ZeroDivisionError, ValueError):
            mutualInfo = 0.0
        try:
            # P(x=0,c=legitimate)
            mutualInfo += (distinctWord.notPresentLegitCount/totalResults) *\
                          math.log((distinctWord.notPresentLegitCount * totalResults) /\
                                     ((distinctWord.notPresentLegitCount+distinctWord.notPresentSpamCount) * (distinctWord.notPresentLegitCount+distinctWord.presentLegitCount)),2)

        except (ZeroDivisionError, ValueError):
            mutualInfo += 0.0
        try:
            # P(x=1,c=spam)
            mutualInfo += (distinctWord.presentSpamCount / totalResults) *\
                          math.log((distinctWord.presentSpamCount * totalResults)/\
                                     ((distinctWord.presentSpamCount+distinctWord.presentLegitCount) * (distinctWord.presentSpamCount + distinctWord.notPresentSpamCount)),2)
        except (ZeroDivisionError, ValueError):
            mutualInfo += 0.0
        try:
            # P(x=1,c=legitimate)
            mutualInfo += (distinctWord.presentLegitCount/totalResults) *\
                          math.log((distinctWord.presentLegitCount * totalResults)/\
                                     ((distinctWord.presentLegitCount+distinctWord.presentSpamCount) * (distinctWord.presentLegitCount + distinctWord.notPresentLegitCount)),2)
        except (ZeroDivisionError, ValueError):
            mutualInfo += 0.0

        return mutualInfo
