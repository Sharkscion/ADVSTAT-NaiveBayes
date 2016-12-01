import os
from controller.Controller import Controller
from mutual_information.EmailReader import EmailReader


testingIndex = 1
controller = Controller()
controller.readEmails('spam emails\\bare\\part',testingIndex)
#controller.collectDistinctWords(testingIndex) #argument: testing index folder ([i+1]th folder); collect all distinct words in the training data excluding the ith folder
controller.selectFeatures()
#controller.computeWordFrequencies()

threshold_lambda = 1
threshold = threshold_lambda/ (1+threshold_lambda)
testingEmailContent = []
s_s = 0 #spam email categorized as spam
s_l = 0 #spam email categorized as legit
l_s = 0 #legit email categorized as spam

for email in controller.partFolderCollection[testingIndex].spamEmail:
    for word in email.split():
        testingEmailContent.append(word)
    result = controller.computeNaiveBayes(testingEmailContent)

    if result > threshold: #isSpam
        s_s+=1
    else: #isLegit
        s_l+=1

for email in controller.partFolderCollection[testingIndex].legitEmail:
    for word in email.split():
        testingEmailContent.append(word)
    result = controller.computeNaiveBayes(testingEmailContent)

    if result > threshold: #isSpam
        l_s+=1

print("Spam Precision: ", s_s/(s_s+l_s))
print("Spam Recall: ",s_s/(s_s+s_l))





'''
partPath = 'C:\\Users\\sharkscion\\Documents\\DLSU Computer Science\\ADVSTAT\\spam emails\\spam emails\\lemm_stop\\part1\\3-1msg3.txt'
content = EmailReader(partPath).read()
emailContent = []
for word in content.split():
    emailContent.append(word)

result = controller.computeNaiveBayes(emailContent,controller.selectFeatures())




'''
