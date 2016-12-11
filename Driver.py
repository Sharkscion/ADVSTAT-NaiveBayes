import os
from controller.Controller import Controller
from mutual_information.EmailReader import EmailReader

controller = Controller()
controller.loadEmails('spam emails\\bare\\part')


threshold_lambda = 999
threshold = threshold_lambda /(1+threshold_lambda)

sPrecision = 0
sRecall = 0
wAcc_b = 0
wErr_b = 0
wAcc = 0
wErr = 0
TCR = 0

for i in range(10):
    testingIndex = i
    controller.preparingTrainingSet(testingIndex)
    controller.selectFeatures()

    # print("Mutual Info top 10.....")
    # for k in controller.trainingDistinctWords:
    #     word = controller.trainingDistinctWords[k]
    #     print(word.content, " MI: ", word.mutualInfo, word.presentSpamCount,
    #           word.notPresentSpamCount, word.presentLegitCount, word.notPresentLegitCount)

    s_s = 0 #spam email categorized as spam
    s_l = 0 #spam email categorized as legit
    l_s = 0 #legit email categorized as spam
    l_l = 0 #legit email categorized as legit

    print("Classifying testing data...[", testingIndex,"]")

    spamSize = len(controller.folderCollection[testingIndex].spamEmail)
    legitSize = len(controller.folderCollection[testingIndex].legitEmail)
    print("testing size:",  spamSize + legitSize)

    for email in controller.folderCollection[testingIndex].spamEmail:
        result = controller.computeNaiveBayes(email)
        if result > threshold: #isSpam
            s_s += 1
        else: #isLegit
            s_l += 1

    for email in controller.folderCollection[testingIndex].legitEmail:
        result = controller.computeNaiveBayes(email)
        if result > threshold: #isSpam
            l_s += 1
        else:
            l_l += 1

    sPrecision += s_s / (s_s + l_s)
    sRecall += s_s / (s_s+s_l)
    wAcc += (threshold_lambda * l_l + s_s)/ (threshold_lambda * legitSize + spamSize)
    wErr += (threshold_lambda * l_s + s_l)/ (threshold_lambda * legitSize + spamSize)
    wAcc_b += (threshold_lambda * legitSize)/(threshold_lambda * legitSize + spamSize)
    wErr_b += spamSize / (threshold_lambda * legitSize + spamSize)
    TCR += wErr_b / wErr


print("AVG Spam Precision: ", sPrecision/10)
print("AVG Spam Precision(1-E): ", 1-wErr/10)
print("AVG Spam Recall: ", sRecall/10)
print("AVG wAcc: ", wAcc/10)
print("AVG wErr: ", wErr/10)
print("AVG wAcc_b: ", wAcc_b/10)
print("AVG wErr_b: ", wErr_b/10)
print("AVG TCR: ", TCR/10)

# print("AVG Spam Precision: ", sPrecision)
# print("AVG Spam Precision(1-E): ", 1-wErr)
# print("AVG Spam Recall: ", sRecall)
# print("AVG wAcc: ", wAcc)
# print("AVG wErr: ", wErr)
# print("AVG wAcc_b: ", wAcc_b)
# print("AVG wErr_b: ", wErr_b)
# print("AVG TCR: ", TCR)


