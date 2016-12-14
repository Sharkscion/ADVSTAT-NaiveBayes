import os
from controller.Controller import Controller
from mutual_information.EmailReader import EmailReader

RECALL = 'recall'
PRECISION = 'precision'

# BARE = 'bare'
# LEMM_STOP = 'lemm_stop'
# LEMM = 'lemm'
# STOP = 'stop'

FILTERS = ['bare', 'lemm', 'stop', 'lemm_stop']

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

tableResults = {}
tableResults[FILTER] = []
tableResults[LAMBDA] = []
tableResults[N_ATTRIBUTES] = []
tableResults[RECALL] = []
tableResults[PRECISION] = []
tableResults[W_ACC] = []
tableResults[BW_ACC] = []
tableResults[TCR] = []


filterCollection = {} #bare, stop, lemm, lemm_stop
controller = Controller()


#in progress: under construction
for i in len(FILTERS):
    controller.loadEmails('spam emails\\'+FILTERS[i]+'\\part')


def runTestTable(filter, threshold, nFeatures):
    path = 'spam emails\\'+filter+'\\part'
    controller.loadEmails(path)

    threshold_lambda = threshold
    threshold = threshold_lambda /(1+threshold_lambda)

    sPrecision = 0
    sRecall = 0
    wAcc_b = 0
    wErr_b = 0
    wAcc = 0
    wErr = 0
    tcr = 0

    for i in range(1):
        testingIndex = i
        controller.preparingTrainingSet(testingIndex)
        controller.selectFeatures(nFeatures)

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

        print("S_S:", s_s)
        print("S_L:", s_l)

        for email in controller.folderCollection[testingIndex].legitEmail:
            result = controller.computeNaiveBayes(email)
            if result > threshold: #isSpam
                l_s += 1
            else:
                l_l += 1

        print("L_S:", l_s)
        print("L_L:", l_l)

        sPrecision += (s_s / (s_s + l_s))
        sRecall += (s_s / (s_s +s_l))
        wAcc += (threshold_lambda * l_l + s_s)/ (threshold_lambda * legitSize + spamSize)
        wErr += (threshold_lambda * l_s + s_l)/ (threshold_lambda * legitSize + spamSize)
        wAcc_b += (threshold_lambda * legitSize)/(threshold_lambda * legitSize + spamSize)
        wErr_b += spamSize / (threshold_lambda * legitSize + spamSize)
        tcr += spamSize / (threshold_lambda*l_s + s_l)

        # print("S_Precision:", sPrecision)
        # print("S_Recall:", sRecall)
        # print("w_acc:", wAcc)
        # print("w_accB:", wAcc_b)
        # print("TCR:", tcr)


    tableResults[FILTER].append(filter)
    tableResults[LAMBDA].append(threshold_lambda)
    tableResults[N_ATTRIBUTES].append(nFeatures)
    tableResults[PRECISION].append((sPrecision/10)*100)#average precision
    tableResults[RECALL].append((sRecall/10)*100) #average recall
    tableResults[W_ACC].append((wAcc/10)*100) #average weighted accuracy
    tableResults[BW_ACC].append((wAcc_b/10)*100) #average baseline weighted accuracy
    tableResults[TCR].append(tcr/10) #average TCR



def runTestPlot(threshold, nFeatures, filterTypes_results):
    path = 'spam emails\\' + filter + '\\part'
    controller.loadEmails(path)
    threshold_lambda = threshold
    threshold = threshold_lambda /(1+threshold_lambda)

    for i in range(10):
        testingIndex = i
        controller.preparingTrainingSet(testingIndex)
        controller.selectFeatures(nFeatures)

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

        filterTypes_results[PRECISION].add(s_s / (s_s + l_s))
        filterTypes_results[RECALL].add(s_s / (s_s +s_l))


runTestTable(BARE,1,50)
# runTestTable(STOP,1,50)
# runTestTable(LEMM,1,100)
# runTestTable(LEMM_STOP,1,100)
#
# runTestTable(BARE,9,200)
# runTestTable(STOP,9,200)
# runTestTable(LEMM,9,100)
# runTestTable(LEMM_STOP,9,100)
#
# runTestTable(BARE,999,200)
# runTestTable(STOP,999,200)
# runTestTable(LEMM,999,300)
# runTestTable(LEMM_STOP,999,300)