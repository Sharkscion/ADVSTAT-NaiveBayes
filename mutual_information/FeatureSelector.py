import math

from model.Word import Word


class FeatureSelector:
    def __init__(self, distinctWords, spamEmails, legitEmails):
        self.distinctWords = distinctWords
        self.spamEmails = spamEmails
        self.legitEmails = legitEmails

    def getWords(self):
       # words = []
        for distinctWord in self.distinctWords:
            distinctWord.mutualInfo = self.getMutualInfo(distinctWord)

        self.distinctWords.sort(key=lambda x: x.mutualInfo, reverse=True)

        return [word.content for word in self.distinctWords][:500]

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