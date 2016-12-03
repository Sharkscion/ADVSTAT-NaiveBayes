#from __future__ import division
import os
import nltk

from model.PartFolder import PartFolder
from model.Word import Word
from mutual_information.EmailReader import EmailReader
from mutual_information.FeatureSelector import FeatureSelector


class Controller:
    SPAM = 'sp'
    LEGITIMATE = "lg"

    def __init__(self):
        self.trainingDistinctWords = []
        self.trainingSpamEmails = []
        self.trainingLegitEmails = []
        self.trainingDistinctWordObjectList = []
        self.folderCollection = []

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
        self.trainingDistinctWords = []
        self.trainingDistinctWordObjectList = []

        ctrSpam = 0
        ctrLegit = 0
        dict = {}

        for i in range(len(self.folderCollection)):
            if i != testingIndex:
                print("Preparing folder[",i,"]...")
                self.trainingSpamEmails += self.folderCollection[i].spamEmail
                self.trainingLegitEmails += self.folderCollection[i].legitEmail

        print("Spam: ", len(self.trainingSpamEmails))
        print("Legit: ", len(self.trainingLegitEmails))
        # for i in range(len(self.folderCollection)):
        #     if i != testingIndex:
        #         print("Preparing folder[",i,"]...")
        for email in self.trainingLegitEmails:
            tokenizedEmail = set(nltk.word_tokenize(email))
            for token in tokenizedEmail:
                if token in dict:
                    word = dict.get(token)
                    word.presentLegitCount += 1
                    word.notPresentLegitCount -= 1
                else:
                    word = Word(token)
                    dict[token] = word
                    word.presentLegitCount = 1
                    word.notPresentLegitCount = len(self.trainingLegitEmails) - 1
                    word.presentSpamCount = 0
                    word.notPresentSpamCount = len(self.trainingSpamEmails)

        for email in self.trainingSpamEmails:
            tokenizedEmail = set(nltk.word_tokenize(email))
            for token in tokenizedEmail:
                if token in dict:
                    word = dict.get(token)
                    word.presentSpamCount += 1
                    word.notPresentSpamCount -= 1
                else:
                    word = Word(token)
                    dict[token] = word
                    word.presentSpamCount = 1
                    word.notPresentSpamCount = len(self.trainingSpamEmails) - 1
                    word.presentLegitCount = 0
                    word.notPresentLegitCount = len(self.trainingLegitEmails)



        self.trainingDistinctWordObjectList = list(dict.values())
        print("Training distinct words: ", len(self.trainingDistinctWordObjectList))

    def selectFeatures(self):
        print("Extracting features/ feature selections..................................")
        fs = FeatureSelector(self.trainingDistinctWordObjectList)
        self.trainingDistinctWordObjectList = fs.getRelevantWords()


    def computeWordObservations(self):
        print("Computing word observations...")
        for word in self.trainingDistinctWordObjectList:
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
        emailContent = nltk.word_tokenize(emailContent)
        probIsSpam = len(self.trainingSpamEmails) / (len(self.trainingSpamEmails) + len(self.trainingLegitEmails))
        probIsLegit = len(self.trainingLegitEmails) / (len(self.trainingSpamEmails) + len(self.trainingLegitEmails))
        # proability relevant words are in the category
        for word in self.trainingDistinctWordObjectList:
            if word.content in emailContent:
                probWord_isPresentSpam *= (word.presentSpamCount/len(self.trainingSpamEmails))
                probWord_isPresentLegit *= (word.presentLegitCount /len(self.trainingLegitEmails))

        return (probIsSpam * probWord_isPresentSpam) / (probIsSpam*probWord_isPresentSpam + probIsLegit*probWord_isPresentLegit)




















