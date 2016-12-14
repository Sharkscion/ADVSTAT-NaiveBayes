#from __future__ import division
import os

import math
from model.PartFolder import PartFolder
from model.Word import Word
from mutual_information.EmailReader import EmailReader
from mutual_information.FeatureSelector import FeatureSelector




class Controller:
    SPAM = 'sp'
    LEGITIMATE = "lg"
    NO_ATTRIBUTE = 50

    def __init__(self):
        self.trainingDistinctWords = {}

    def loadEmails(self, path):
        print("Loading emails...")

        folderCollection = []

        for i in range(1,11):

            partPath = path + str(i)
            partFolder = PartFolder()

            for filename in os.listdir(partPath):
                content = EmailReader(partPath + '\\' + filename).read()
                if filename.startswith(self.SPAM):
                    partFolder.spamEmail.append(content)
                else:
                    partFolder.legitEmail.append(content)

            folderCollection.append(partFolder)

        for i in range(10): #testingIndex
            for j in range(10):
                if j != i:
                    folderCollection[i].trainingSpamEmail += folderCollection[j].spamEmail
                    folderCollection[i].trainingLegitEmail += folderCollection[j].legitEmail


        for i in range(10):
            folderCollection[i].relevantWords = self.praparingTrainingSet(folderCollection[i])


    def praparingTrainingSet(self, testingFolder):

        trainingDistinctWords = {}

        for email in testingFolder.trainingLegitEmails:
            email = email.split()
            tokenizedEmail = set(email)

            # count term frequencies
            for token in tokenizedEmail:
                if token in trainingDistinctWords:
                    word = trainingDistinctWords.get(token)
                    word.presentLegitCount += 1
                    word.notPresentLegitCount -= 1
                else:
                    word = Word(token)
                    word.presentLegitCount = 1
                    word.notPresentLegitCount = len(testingFolder.trainingLegitEmails) - 1
                    word.presentSpamCount = 0
                    word.notPresentSpamCount = len(testingFolder.trainingSpamEmails)
                    trainingDistinctWords[token] = word

        for email in testingFolder.trainingSpamEmails:
            email = email.split()
            tokenizedEmail = set(email)
            for token in tokenizedEmail:
                if token in trainingDistinctWords:
                    word = trainingDistinctWords.get(token)
                    word.presentSpamCount += 1
                    word.notPresentSpamCount -= 1
                else:
                    word = Word(token)
                    word.presentSpamCount = 1
                    word.notPresentSpamCount = len(testingFolder.trainingSpamEmails) - 1
                    word.presentLegitCount = 0
                    word.notPresentLegitCount = len(testingFolder.trainingLegitEmails)
                    trainingDistinctWords[token] = word

        fs = FeatureSelector(trainingDistinctWords)
        return fs.getRelevantWords()

    #
    # def preparingTrainingSet(self, testingIndex):
    #
    #     print("Preparing training set (find distinct words and load training spam and legit emails)...")
    #     self.trainingSpamEmails = []
    #     self.trainingLegitEmails = []
    #     self.trainingDistinctWords = {}
    #
    #     for i in range(len(self.folderCollection)):
    #         if i != testingIndex:
    #             self.trainingSpamEmails += self.folderCollection[i].spamEmail
    #             self.trainingLegitEmails += self.folderCollection[i].legitEmail
    #
    #     for email in self.trainingLegitEmails:
    #         email = email.split()
    #         tokenizedEmail = set(email)
    #
    #         #count term frequencies
    #         for token in tokenizedEmail:
    #             if token in self.trainingDistinctWords:
    #                 word = self.trainingDistinctWords.get(token)
    #                 word.presentLegitCount += 1
    #                 word.notPresentLegitCount -= 1
    #             else:
    #                 word = Word(token)
    #                 word.presentLegitCount = 1
    #                 word.notPresentLegitCount = len(self.trainingLegitEmails) - 1
    #                 word.presentSpamCount = 0
    #                 word.notPresentSpamCount = len(self.trainingSpamEmails)
    #                 self.trainingDistinctWords[token] = word
    #
    #
    #
    #     for email in self.trainingSpamEmails:
    #         email = email.split()
    #         tokenizedEmail = set(email)
    #         for token in tokenizedEmail:
    #             if token in self.trainingDistinctWords:
    #                 word = self.trainingDistinctWords.get(token)
    #                 word.presentSpamCount += 1
    #                 word.notPresentSpamCount -= 1
    #             else:
    #                 word = Word(token)
    #                 word.presentSpamCount = 1
    #                 word.notPresentSpamCount = len(self.trainingSpamEmails) - 1
    #                 word.presentLegitCount = 0
    #                 word.notPresentLegitCount = len(self.trainingLegitEmails)
    #                 self.trainingDistinctWords[token] = word
    #
    #
    #
    #     print("Training distinct words: ", len(self.trainingDistinctWords))
    #
    # #compute for all mutual information
    # def findRelevantWords(self):
    #     for i in range(10):
    #         print("Gettting relevant words per part folder...[",i,"]")



    def selectNFeatures(self, nFeatures, testingFolder):
        print("Extracting features/ feature selections...")

        self.trainingDistinctWords = {x[0]: x[1] for x in testingFolder.relevantWords[:nFeatures]}

        for key in self.trainingDistinctWords:
            word = self.trainingDistinctWords[key]

            for email in testingFolder.trainingSpamEmails:
                if word.content in email.split():
                    word.spamDocumentCount += 1 #count document frequencies

            for email in testingFolder.trainingLegitEmails:
                if word.content in email.split():
                    word.legitDocumentCount += 1 #count document frequencies


    def computeNaiveBayes(self, testingFolder, emailContent):
        #Naive Bayes: Multinomial NB, TF attributes

        emailContent = emailContent.split()
        dict_testingData = {} # dictionary of distinct words in testing data
        total_trainingEmails = len(testingFolder.trainingSpamEmails) + len(testingFolder.trainingLegitEmails)
        probIsSpam = len(testingFolder.trainingSpamEmails) / total_trainingEmails
        probIsLegit = len(testingFolder.trainingLegitEmails) / total_trainingEmails
        probWord_isPresentSpam = 1.0
        probWord_isPresentLegit = 1.0

        #determine whther term appeared in document
        for key in self.trainingDistinctWords:
            if key in emailContent:
                dict_testingData[key] = 1
            else:
                dict_testingData[key] = 0

        for key in self.trainingDistinctWords:
            word = self.trainingDistinctWords[key]
            power = dict_testingData[key]

            prob_t_s = (1 + word.spamDocumentCount) / (2 + len(testingFolder.trainingSpamEmails))
            prob_t_l = (1 + word.legitDocumentCount) / (2 + len(testingFolder.trainingLegitEmails))

            probWord_isPresentSpam*= (math.pow(prob_t_s, power) * math.pow(1-prob_t_s, 1-power))
            probWord_isPresentLegit*= (math.pow(prob_t_l, power) * math.pow(1-prob_t_l, 1-power))

        return (probIsSpam * probWord_isPresentSpam) / (probIsSpam * probWord_isPresentSpam + probIsLegit * probWord_isPresentLegit)

        # #emailContent = nltk.word_tokenize(emailContent)



        # proability relevant words are in the category
        # for word in emailContent:
        #     if word in self.trainingDistinctWords:
        #         relWord = self.trainingDistinctWords[word]
        #         if relWord.presentSpamCount != 0:
        #             probWord_isPresentSpam *= (relWord.presentSpamCount / len(self.trainingSpamEmails))
        #
        #         if relWord.presentLegitCount != 0:
        #             probWord_isPresentLegit *= (relWord.presentLegitCount / len(self.trainingLegitEmails))


        # proability relevant words are in the category
        # for k in self.trainingDistinctWords:
        #     word = self.trainingDistinctWords[k]
        #     if word.content in emailContent:
        #         if word.presentSpamCount != 0:
        #             probWord_isPresentSpam *= (word.presentSpamCount/self.nWordsSpam)
        #
        #         if word.presentLegitCount != 0:
        #             probWord_isPresentLegit *= (word.presentLegitCount/self.nWordsLegit)

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

