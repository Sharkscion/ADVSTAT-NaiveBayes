import math
import operator
from model.Word import Word


class FeatureSelector:
    def __init__(self, distinctWords):
        self.distinctWords = distinctWords

    def getRelevantWords(self):
        for k in self.distinctWords:
            self.distinctWords[k].mutualInfo = self.getMutualInfo(self.distinctWords[k])

        self.distinctWords = sorted(self.distinctWords.items(), key=lambda x:x[1].mutualInfo, reverse = True)[:50]
        self.distinctWords = {x[0]: x[1] for x in self.distinctWords}
        # print("after convert back to dictionary..........................")
        # for k in self.distinctWords:
        #     print(self.distinctWords.get(k).content , ": ",self.distinctWords[k].mutualInfo)

        return self.distinctWords


    def getMutualInfo(self, distinctWord):

        totalResults = distinctWord.presentSpamCount + distinctWord.notPresentSpamCount + \
                       distinctWord.presentLegitCount + distinctWord.notPresentLegitCount

        try:
            # P(x=0,c=spam)
            mutualInfo = (distinctWord.notPresentSpamCount/totalResults)*\
                         math.log10((distinctWord.notPresentSpamCount * totalResults)/\
                                    ((distinctWord.notPresentSpamCount+distinctWord.notPresentLegitCount)*(distinctWord.notPresentSpamCount+distinctWord.presentSpamCount)))

        except (ZeroDivisionError, ValueError):
            mutualInfo = 0.0
        try:
            # P(x=0,c=legitimate)
            mutualInfo += (distinctWord.notPresentLegitCount/totalResults) *\
                          math.log10((distinctWord.notPresentLegitCount * totalResults) /\
                                     ((distinctWord.notPresentLegitCount+distinctWord.notPresentSpamCount) * (distinctWord.notPresentLegitCount+distinctWord.presentLegitCount)))

        except (ZeroDivisionError, ValueError):
            mutualInfo += 0.0
        try:
            # P(x=1,c=spam)
            mutualInfo += (distinctWord.presentSpamCount / totalResults) *\
                          math.log10((distinctWord.presentSpamCount * totalResults)/\
                                     ((distinctWord.presentSpamCount+distinctWord.presentLegitCount) * (distinctWord.presentSpamCount + distinctWord.notPresentSpamCount)))
        except (ZeroDivisionError, ValueError):
            mutualInfo += 0.0
        try:
            # P(x=1,c=legitimate)
            mutualInfo += (distinctWord.presentLegitCount/totalResults) *\
                          math.log10((distinctWord.presentLegitCount * totalResults)/\
                                     ((distinctWord.presentLegitCount+distinctWord.presentSpamCount) * (distinctWord.presentLegitCount + distinctWord.notPresentLegitCount)))
        except (ZeroDivisionError, ValueError):
            mutualInfo += 0.0

        return mutualInfo
