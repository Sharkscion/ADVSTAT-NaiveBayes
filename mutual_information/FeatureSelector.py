import math

from model.Word import Word


class FeatureSelector:
    def __init__(self, distinctWords, spamEmails, legitEmails):
        self.distinctWords = distinctWords
        self.spamEmails = spamEmails
        self.legitEmails = legitEmails

    def getRelevantWords(self):
       # words = []


        print("Getting 500 relevant words (Mutual Information)......................")

       # self.computeWordFrequencies()
        i = 0
        for distinctWord in self.distinctWords:
            print("distinct word [", i, "] : ", distinctWord.content)
            self.computePresentSpamCount(distinctWord)
            self.computeNotPresentSpamCount(distinctWord)
            self.computePresentLegitCount(distinctWord)
            self.computeNotPresentLegitCount(distinctWord)
            distinctWord.mutualInfo = self.getMutualInfo(distinctWord)
            i+=1

        self.distinctWords.sort(key=lambda x: x.mutualInfo, reverse=True)
        print("Finish getting 500 relevant words (Mutual Information)......................")

        return [word.content for word in self.distinctWords][:500]


    def computePresentSpamCount(self, distinctWord):
        distinctWord.presentSpamCount = 0
        for spamEmail in self.spamEmails:
            if distinctWord.content in spamEmail.split():
                distinctWord.presentSpamCount += 1


    def computeNotPresentSpamCount(self, distinctWord):
        distinctWord.notPresentSpamCount = len(self.spamEmails) - distinctWord.presentSpamCount


    def computePresentLegitCount(self, distinctWord):
        distinctWord.presentLegitCount = 0
        for legitEmail in self.legitEmails:
            if distinctWord.content in legitEmail.split():
                distinctWord.presentLegitCount += 1

    def computeNotPresentLegitCount(self, distinctWord):
        distinctWord.notPresentLegitCount = len(self.legitEmails) - distinctWord.presentLegitCount

    def getMutualInfo(self, distinctWord):
        mutualInfo = 0

        presentCount = distinctWord.presentSpamCount + distinctWord.presentLegitCount

        totalEmails = len(self.spamEmails) + len(self.legitEmails)
        notPresentCount = totalEmails - presentCount
        '''
        print('WORD:', distinctWord)
        print('PS:' , distinctWord.presentSpamCount)
        print('NPS:', distinctWord.notPresentSpamCount)
        print('SE:', len(self.spamEmails))
        print('NP:', notPresentCount)
        print('TE:', totalEmails)
        '''
        try:
            # P(x=0,c=spam)
            mutualInfo = (distinctWord.notPresentSpamCount / len(self.spamEmails)) * \
                         math.log10((distinctWord.notPresentSpamCount / len(self.spamEmails)) / (
                         (notPresentCount / totalEmails) * (len(self.spamEmails) / totalEmails)))
        except (ZeroDivisionError, ValueError):
            mutualInfo = 0

        try:
            # P(x=0,c=legitimate)
            mutualInfo += (distinctWord.notPresentLegitCount / len(self.legitEmails)) * \
                          math.log10((distinctWord.notPresentLegitCount / len(self.legitEmails)) / (
                          (notPresentCount / totalEmails) * (len(self.legitEmails) / totalEmails)))
        except (ZeroDivisionError, ValueError):
            mutualInfo += 0

        try:
            # P(x=1,c=spam)
            mutualInfo += (distinctWord.presentSpamCount / len(self.spamEmails)) * \
                          math.log10((distinctWord.presentSpamCount / len(self.spamEmails)) / (
                              (presentCount / totalEmails) * (len(self.spamEmails) / totalEmails)))
        except (ZeroDivisionError, ValueError):
            mutualInfo += 0

        try:
            # P(x=1,c=legitimate)
            mutualInfo += (distinctWord.presentLegitCount / len(self.legitEmails)) * \
                          math.log10((distinctWord.presentLegitCount / len(self.legitEmails)) / (
                          (presentCount / totalEmails) * (len(self.legitEmails) / totalEmails)))
        except (ZeroDivisionError, ValueError):
            mutualInfo += 0

        return mutualInfo