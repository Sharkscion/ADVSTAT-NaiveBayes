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

