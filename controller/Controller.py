#from __future__ import division
import os

from model.Word import Word
from mutual_information.EmailReader import EmailReader
from mutual_information.FeatureSelector import FeatureSelector


class Controller:
    SPAM = 'sp'
    LEGITIMATE = "lg"

    def __init__(self):
        self.distinctWords = []
        self.spamEmails = []
        self.legitEmails = []
        self.distinctWordObjectList = []

    def readEmails(self, path):

        print("Getting distinct words..................................")
        for i in range(2,11):
            partPath = path + str(i)
            for filename in os.listdir(partPath):
                content = EmailReader(partPath + '\\' + filename).read()

                for word in content.split():
                    if word not in self.distinctWords:
                        self.distinctWords.append(word)

                if filename.startswith(self.SPAM):
                    self.spamEmails.append(content)
                else:
                    self.legitEmails.append(content)

        for word in self.distinctWords:
            self.distinctWordObjectList.append(Word(word,0))

        print("Finish getting distinct words..................................")

    def selectFeatures(self):
        print("Extracting features/ feature selections..................................")
        fs = FeatureSelector(self.distinctWordObjectList, self.spamEmails, self.legitEmails)
        return fs.getWords()


    def computeWordFrequencies(self):
        print("Computing word frequencies..................................")
        for word in self.distinctWordObjectList:
            self.computePresentSpamCount(word)
            self.computeNotPresentSpamCount(word)
            self.computePresentLegitCount(word)
            self.computeNotPresentLegitCount(word)
        print("Finish computing word frequencies..................................")

    def computePresentSpamCount(self, distinctWord):
        distinctWord.presentSpamCount = 0
        for spamEmail in self.spamEmails:
            if distinctWord.content in spamEmail:
                distinctWord.presentSpamCount += 1

    def computeNotPresentSpamCount(self, distinctWord):
        distinctWord.notPresentSpamCount = len(self.spamEmails) - distinctWord.presentSpamCount


    def computePresentLegitCount(self, distinctWord):
        distinctWord.presentLegitCount = 0
        for legitEmail in self.legitEmails:
            if distinctWord.content in legitEmail:
                distinctWord.presentLegitCount += 1

    def computeNotPresentLegitCount(self, distinctWord):
        distinctWord.notPresentLegitCount = len(self.legitEmails) - distinctWord.presentLegitCount


    def computeNaiveBayes(self, emailContent, relevantWords):
        print("Calculating Naive Bayes..................................")

        relevantWords = relevantWords[:500]
        probWord_isSpam = 1.0
        probWord_isLegit = 1.0


        probIsSpam = len(self.spamEmails) / (len(self.spamEmails) + len(self.legitEmails))

        print("probIsSpam: ", probIsSpam)

        # proability relevant words are in the category
        for word in relevantWords:
            if word.content in emailContent:
                probWord_isSpam *= word.presentSpamCount / len(self.spamEmails)
                probWord_isLegit *= word.presentLegitCount / len(self.legitEmails)
                #print("probWord_isSpam variable: ", probWord_isSpam)
                #print("probWord_isSpam variable: ", probWord_isLegit)


        probAllRelevantWordsOccured = probWord_isSpam + probWord_isLegit

        #print("ProbAllRelevantWords Occured:", probAllRelevantWordsOccured)
        return probIsSpam * probWord_isSpam / probAllRelevantWordsOccured

















