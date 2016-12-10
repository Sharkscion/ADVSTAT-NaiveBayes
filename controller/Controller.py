#from __future__ import division
import os

import math
import nltk
from model.PartFolder import PartFolder
from model.Word import Word
from mutual_information.EmailReader import EmailReader
from mutual_information.FeatureSelector import FeatureSelector


class Controller:
    SPAM = 'sp'
    LEGITIMATE = "lg"

    def __init__(self):
        self.trainingDistinctWords = {}
        self.trainingSpamEmails = []
        self.trainingLegitEmails = []
        #self.trainingDistinctWordObjectList = []
        self.folderCollection = []

        self.nWordsSpam = 0
        self.nWordsLegit = 0

    def loadEmails(self, path):
        print("Loading emails...")
        for i in range(1,11):
            partPath = path + str(i)
            partFolder = PartFolder()
            print("Loading email[",i,"]...")
            for filename in os.listdir(partPath):
                content = EmailReader(partPath + '\\' + filename).read()
                if filename.startswith(self.SPAM):
                    partFolder.addSpamEmail(content)
                else:
                    partFolder.addLegitEmail(content)

            self.folderCollection.append(partFolder)

    def preparingTrainingSet(self, testingIndex):

        print("Preparing training set (find distinct words and load training spam and legit emails)...")
        self.trainingSpamEmails = []
        self.trainingLegitEmails = []
        self.trainingDistinctWords = {}
        self.nWordsSpam = 0
        self.nWordsLegit = 0
        #self.trainingDistinctWordObjectList = []

        ctrSpam = 0
        ctrLegit = 0
        #dict = {}

        for i in range(len(self.folderCollection)):
            if i != testingIndex:
                print("Preparing folder[",i,"]...")
                self.trainingSpamEmails += self.folderCollection[i].spamEmail
                self.trainingLegitEmails += self.folderCollection[i].legitEmail

        print("Spam: ", len(self.trainingSpamEmails))
        print("Legit: ", len(self.trainingLegitEmails))

        for email in self.trainingLegitEmails:
            email = email.split()
            self.nWordsLegit += len(email)
            tokenizedEmail = set(email)
            for token in tokenizedEmail:
                if token in self.trainingDistinctWords:
                    word = self.trainingDistinctWords.get(token)
                    word.presentLegitCount += 1
                    word.notPresentLegitCount -= 1
                else:
                    word = Word(token)
                    word.presentLegitCount = 1
                    word.notPresentLegitCount = len(self.trainingLegitEmails) - 1
                    word.presentSpamCount = 0
                    word.notPresentSpamCount = len(self.trainingSpamEmails)
                    self.trainingDistinctWords[token] = word

        for email in self.trainingSpamEmails:
            email = email.split()
            self.nWordsSpam += len(email)
            tokenizedEmail = set(email)
            for token in tokenizedEmail:
                if token in self.trainingDistinctWords:
                    word = self.trainingDistinctWords.get(token)
                    word.presentSpamCount += 1
                    word.notPresentSpamCount -= 1
                else:
                    word = Word(token)
                    word.presentSpamCount = 1
                    word.notPresentSpamCount = len(self.trainingSpamEmails) - 1
                    word.presentLegitCount = 0
                    word.notPresentLegitCount = len(self.trainingLegitEmails)
                    self.trainingDistinctWords[token] = word


        print("Training distinct words: ", len(self.trainingDistinctWords))

    def selectFeatures(self):
        print("Extracting features/ feature selections..................................")
        fs = FeatureSelector(self.trainingDistinctWords)
        self.trainingDistinctWords = fs.getRelevantWords()



    def computeWordObservations(self):
        print("Computing word observations...")
        for k in self.trainingDistinctWords:
            word =  self.trainingDistinctWords[k]
            self.computePresentSpamCount(word)
            self.computeNotPresentSpamCount(word)
            self.computePresentLegitCount(word)
            self.computeNotPresentLegitCount(word)
        print("Finish computing word observations...")

    def computePresentSpamCount(self, distinctWord):
        distinctWord.presentSpamCount = 0
        for spamEmail in self.trainingSpamEmails:
            if distinctWord.content in spamEmail:
                distinctWord.presentSpamCount += 1

    def computeNotPresentSpamCount(self, distinctWord):
        distinctWord.notPresentSpamCount = len(self.trainingSpamEmails) - distinctWord.presentSpamCount


    def computePresentLegitCount(self, distinctWord):
        distinctWord.presentLegitCount = 0
        for legitEmail in self.trainingLegitEmails:
            if distinctWord.content in legitEmail:
                distinctWord.presentLegitCount += 1

    def computeNotPresentLegitCount(self, distinctWord):
        distinctWord.notPresentLegitCount = len(self.trainingLegitEmails) - distinctWord.presentLegitCount


    def computeNaiveBayes(self, emailContent):
        probWord_isPresentSpam = 1.0
        probWord_isPresentLegit = 1.0
        #emailContent = nltk.word_tokenize(emailContent)
        probWord_isPresentSpamLog = 0
        probWord_isPresentLegitLog = 0
        emailContent = emailContent.split()
        probIsSpam = len(self.trainingSpamEmails) / (len(self.trainingSpamEmails) + len(self.trainingLegitEmails))
        probIsLegit = len(self.trainingLegitEmails) / (len(self.trainingSpamEmails) + len(self.trainingLegitEmails))

        # proability relevant words are in the category
        # for word in emailContent:
        #     if word in self.trainingDistinctWords:
        #         relWord = self.trainingDistinctWords[word]
        #         probWord_isPresentSpam *= (relWord.presentSpamCount / len(self.trainingSpamEmails))
        #         probWord_isPresentLegit *= (relWord.presentLegitCount / len(self.trainingLegitEmails))

        # proability relevant words are in the category
        for k in self.trainingDistinctWords:
            word = self.trainingDistinctWords[k]
            if word.content in emailContent:
                if word.presentSpamCount != 0:
                    probWord_isPresentSpam *= (word.presentSpamCount/self.nWordsSpam)

                if word.presentLegitCount != 0:
                    probWord_isPresentLegit *= (word.presentLegitCount/self.nWordsLegit)

        # for k in self.trainingDistinctWords:
        #     word = self.trainingDistinctWords[k]
        #     if word.content in emailContent:
        #         if word.presentSpamCount != 0:
        #             probWord_isPresentSpamLog += math.log10((word.presentSpamCount / len(self.trainingSpamEmails)))
        #             probWord_isPresentSpam *= (word.presentSpamCount / len(self.trainingSpamEmails))
        #
        #         if word.presentLegitCount != 0:
        #             probWord_isPresentLegitLog += math.log10((word.presentLegitCount/len(self.trainingLegitEmails)))
        #             probWord_isPresentLegit *= (word.presentLegitCount/len(self.trainingLegitEmails))
        #
        # rightValue = (math.log10(probIsSpam) + probWord_isPresentSpamLog) - math.log10(probIsSpam * probWord_isPresentSpam + probIsLegit * probWord_isPresentLegit)
        # #return 10 ** rightValue
        return (probIsSpam * probWord_isPresentSpam) / (probIsSpam*probWord_isPresentSpam + probIsLegit*probWord_isPresentLegit)
