import os
import math

from model.Result import Result
from model.PartFolder import PartFolder
from model.Word import Word
from mutual_information.FeatureSelector import FeatureSelector

RECALL = 'recall'
PRECISION = 'precision'

FILTERS = ['bare', 'stop', 'lemm', 'lemm_stop']
LAMBDA = 'lambda'
N_ATTRIBUTES = 'attribute'
W_ACC = 'w_acc'
BW_ACC = 'bw_acc'
TCR = 'tcr'
FILTER = 'filter'

bare_resultsPR = {}
bare_resultsPR[PRECISION] = []
bare_resultsPR[RECALL] = []

lemm_resultsPR = {}
lemm_resultsPR[PRECISION] = []
lemm_resultsPR[RECALL] = []

stop_resultsPR = {}
stop_resultsPR[PRECISION] = []
stop_resultsPR[RECALL] = []

lemm_stop_resultsPR = {}
lemm_stop_resultsPR[PRECISION] = []
lemm_stop_resultsPR[RECALL] = []

tableResults = []
filterCollection = {} #bare, stop, lemm, lemm_stop


def loadEmails(path):
    print("Loading emails...:",path)

    folderCollection = []

    #pre-load the emails of each folder per filter configuration
    for i in range(1, 11):
        partPath = path + str(i)
        partFolder = PartFolder()

        for filename in os.listdir(partPath):
            content = open(partPath + '\\' + filename).read()
            if filename.startswith('sp'):
                partFolder.spamEmail.append(content)
            else:
                partFolder.legitEmail.append(content)

        folderCollection.append(partFolder)

    #pre-load the training emails classified as spam and legit
    for i in range(10):  # testingIndex
        for j in range(10):
            if j != i:
                folderCollection[i].trainingSpamEmail += folderCollection[j].spamEmail
                folderCollection[i].trainingLegitEmail += folderCollection[j].legitEmail

    #pre-load and find the relevant words per testing index
    for i in range(10):
        folderCollection[i].relevantWords = praparingTrainingSet(folderCollection[i])

    return folderCollection

def praparingTrainingSet(testingFolder):

    trainingDistinctWords = {}

    for email in testingFolder.trainingLegitEmail:
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
                word.notPresentLegitCount = len(testingFolder.trainingLegitEmail) - 1
                word.presentSpamCount = 0
                word.notPresentSpamCount = len(testingFolder.trainingSpamEmail)
                trainingDistinctWords[token] = word

    for email in testingFolder.trainingSpamEmail:
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
                word.notPresentSpamCount = len(testingFolder.trainingSpamEmail) - 1
                word.presentLegitCount = 0
                word.notPresentLegitCount = len(testingFolder.trainingLegitEmail)
                trainingDistinctWords[token] = word

    fs = FeatureSelector(trainingDistinctWords)
    return fs.getRelevantWords()


def selectNFeatures(nFeatures, testingFolder):
    relevantWords = {x[0]: x[1] for x in testingFolder.relevantWords[:nFeatures]}

    for key in relevantWords:
        word = relevantWords[key]

        for email in testingFolder.trainingSpamEmail:
            if word.content in email.split():
                word.spamDocumentCount += 1  # count document frequencies

        for email in testingFolder.trainingLegitEmail:
            if word.content in email.split():
                word.legitDocumentCount += 1  # count document frequencies

    return relevantWords

def computeNaiveBayes(testingFolder, emailContent, relevantWords):
    # Naive Bayes: Multinomial NB, TF attributes
    emailContent = emailContent.split()
    dict_testingData = {}  # dictionary of distinct words in testing data

    total_trainingEmails = len(testingFolder.trainingSpamEmail) + len(testingFolder.trainingLegitEmail)

    probIsSpam = len(testingFolder.trainingSpamEmail) / total_trainingEmails
    probIsLegit = len(testingFolder.trainingLegitEmail) / total_trainingEmails

    probWord_isPresentSpam = 1.0
    probWord_isPresentLegit = 1.0

    # determine whther term appeared in document
    for key in relevantWords:
        if key in emailContent:
            dict_testingData[key] = 1
        else:
            dict_testingData[key] = 0

    for key in relevantWords:
        word = relevantWords[key]
        power = dict_testingData[key]

        prob_t_s = (1 + word.spamDocumentCount) / (2 + len(testingFolder.trainingSpamEmail))
        prob_t_l = (1 + word.legitDocumentCount) / (2 + len(testingFolder.trainingLegitEmail))

        probWord_isPresentSpam *= (math.pow(prob_t_s, power) * math.pow(1 - prob_t_s, 1 - power))
        probWord_isPresentLegit *= (math.pow(prob_t_l, power) * math.pow(1 - prob_t_l, 1 - power))

    return (probIsSpam * probWord_isPresentSpam) / ( probIsSpam * probWord_isPresentSpam + probIsLegit * probWord_isPresentLegit)


#function for constructing the average results table per filter, nAttributes, and threshold configuration
def runTestTable(filter, threshold, nFeatures):

    folderCollection = filterCollection[filter]
    threshold_lambda = threshold
    threshold = threshold_lambda /(1+threshold_lambda)


    sPrecision = 0
    sRecall = 0
    wAcc_b = 0
    wErr_b = 0
    wAcc = 0
    wErr = 0
    tcr = 0

    for testingIndex in range(10):
        relevantWords = selectNFeatures(nFeatures, folderCollection[testingIndex])

        s_s = 0 #spam email categorized as spam
        s_l = 0 #spam email categorized as legit
        l_s = 0 #legit email categorized as spam
        l_l = 0 #legit email categorized as legit

        spamSize = len(folderCollection[testingIndex].spamEmail)
        legitSize = len(folderCollection[testingIndex].legitEmail)

        for email in folderCollection[testingIndex].spamEmail:
            result = computeNaiveBayes(folderCollection[testingIndex], email, relevantWords)
            if result > threshold: #isSpam
                s_s += 1
            else: #isLegit
                s_l += 1


        for email in folderCollection[testingIndex].legitEmail:
            result = computeNaiveBayes(folderCollection[testingIndex], email, relevantWords)
            if result > threshold: #isSpam
                l_s += 1
            else:
                l_l += 1

        sPrecision += (s_s / (s_s + l_s))
        sRecall += (s_s / (s_s +s_l))
        wAcc += (threshold_lambda * l_l + s_s)/ (threshold_lambda * legitSize + spamSize)
        wErr += (threshold_lambda * l_s + s_l)/ (threshold_lambda * legitSize + spamSize)
        wAcc_b += (threshold_lambda * legitSize)/(threshold_lambda * legitSize + spamSize)
        wErr_b += spamSize / (threshold_lambda * legitSize + spamSize)
        tcr += spamSize / (threshold_lambda*l_s + s_l)


    table_row = Result()
    table_row.filter = filter
    table_row.threshold = threshold_lambda
    table_row.nFeatures = nFeatures
    table_row.avg_recall =  (sRecall/10)*100
    table_row.avg_precision = (sPrecision/10)*100
    table_row.avg_w_acc = (wAcc/10)*100
    table_row.avg_bw_acc = (wAcc_b/10)*100
    table_row.avg_tcr = tcr/10
    tableResults.append(table_row)

    print("S_Precision:", (sPrecision/10)*100)
    print("S_Recall:", (sRecall/10)*100)
    print("w_acc:", (wAcc/10)*100)
    print("w_accB:", (wAcc_b/10)*100)
    print("TCR:", tcr/10)


def table_row_generator(index, pd, filter, threshold, nFeatures, avg_recall, avg_precision, avg_accuracy, avg_accuracy_base,
                        avg_tcr):
    raw_data = {
        '#': index,
        'Filter Configuration': [filter],
        'Lambda': [threshold],
        'No. of attrib.': [nFeatures],
        'Spam Recall': [avg_recall],
        'Spam Precision': [avg_precision],
        'Weighted Accuracy': [avg_accuracy],
        'Baseline W. Acc': [avg_accuracy_base],
        'TCR': [avg_tcr]
    }
    return pd.DataFrame(raw_data, columns=['#', 'Filter Configuration', 'Lambda', 'No. of attrib.', 'Spam Recall',
                                           'Spam Precision', 'Weighted Accuracy', 'Baseline W. Acc', 'TCR'])



#in progress: under construction
for i in range(len(FILTERS)):
    filterCollection[FILTERS[i]] = loadEmails('spam emails\\'+FILTERS[i]+'\\part')

runTestTable(FILTERS[0], 1, 50)
runTestTable(FILTERS[1], 1, 50)





