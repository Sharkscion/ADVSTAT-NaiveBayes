import os
from controller.Controller import Controller
from mutual_information.EmailReader import EmailReader


controller = Controller()
controller.readEmails('spam emails\\lemm_stop\\part')
controller.computeWordFrequencies()
controller.selectFeatures()
controller.saveWords()
#controller.computeWordFrequencies()

'''
partPath = 'C:\\Users\\sharkscion\\Documents\\DLSU Computer Science\\ADVSTAT\\spam emails\\spam emails\\lemm_stop\\part1\\3-1msg3.txt'
content = EmailReader(partPath).read()
emailContent = []
for word in content.split():
    emailContent.append(word)

result = controller.computeNaiveBayes(emailContent,controller.selectFeatures())

threshold_lambda = 1
threshold = threshold_lambda/ (1+threshold_lambda)

if result > threshold:
    print("IS SPAM: ", result, " > ", threshold)
else:
    print("IS LEGIT: ", result, " < ", threshold)
'''
