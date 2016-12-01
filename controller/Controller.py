#from __future__ import division
import os

from model.FileIO import FileIO
from model.PartFolder import Part
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
        self.partFolderCollection = []

    def readEmails(self, path, testingIndex):

        print("Loading emails..................................")
        for i in range(1,11):
            partPath = path + str(i)
            partFolder = Part(partPath)
            print("loading folder[", i, "]..........")
            for filename in os.listdir(partPath):
                content = EmailReader(partPath + '\\' + filename).read()


                if i != testingIndex:
                    for word in content.split():
                        if word not in self.distinctWords:
                            self.distinctWords.append(word)

                if filename.startswith(self.SPAM):
                    self.spamEmails.append(content)
                    partFolder.addSpamEmail(content)
                else:
                    self.legitEmails.append(content)
                    partFolder.addLegitEmail(content)

            self.partFolderCollection.append(partFolder)

        print("Distinct words size: ", len(self.distinctWords))
        print("Finish loading emails..................................")

    # def collectDistinctWords(self, testingIndex):
    #
    #     print("Size part folder collection: ", len(self.partFolderCollection))
    #     print("Collecting distinct words in training data.............................")
    #     for i in range(len(self.partFolderCollection)):
    #         if(i != testingIndex):
    #             for email in self.partFolderCollection[i].spamEmail:
    #                 for word in email.split():
    #                     if word not in self.distinctWords:
    #                         self.distinctWords.append(word)
    #
    #             for email in self.partFolderCollection[i].legitEmail:
    #                 for word in email.split():
    #                     if word not in self.distinctWords:
    #                         self.distinctWords.append(word)
    #
    #     print("Finish collecting distinct words in training data.............................")
    #     print("Number of distinct words: ", len(self.distinctWords))

    # def saveWords(self):
    #     FileIO().writeWords(self.distinctWords)

    # def loadWords(self):
    #     self.distinctWords = FileIO().readWords()
    #     self.computeWordFrequencies()

    def selectFeatures(self):
        print("Extracting features/ feature selections..................................")
        self.distinctWordObjectList = [Word(word, 0) for word in self.distinctWords]
        fs = FeatureSelector(self.distinctWordObjectList, self.spamEmails, self.legitEmails)
        self.distinctWords = fs.getRelevantWords()
        print("Finish extracting features/ feature selections..................................")


    def computeNaiveBayes(self, emailContent):
        print("Calculating Naive Bayes..................................")
        probWord_isSpam = 1.0
        probWord_isLegit = 1.0

        probIsSpam = len(self.spamEmails) / (len(self.spamEmails) + len(self.legitEmails))

        print("probIsSpam: ", probIsSpam)

        # proability relevant words are in the category
        for word in self.distinctWords:
            if word.content in emailContent:
                probWord_isSpam *= word.presentSpamCount / len(self.spamEmails)
                probWord_isLegit *= word.presentLegitCount / len(self.legitEmails)
                #print("probWord_isSpam variable: ", probWord_isSpam)
                #print("probWord_isSpam variable: ", probWord_isLegit)


        probAllRelevantWordsOccured = probWord_isSpam + probWord_isLegit

        #print("ProbAllRelevantWords Occured:", probAllRelevantWordsOccured)
        return probIsSpam * probWord_isSpam / probAllRelevantWordsOccured

















