class Word:
    def __init__(self, content, mutualInfo):
        self.content = content
        self.mutualInfo = mutualInfo
        self.notPresentSpamCount = 0;
        self.notPresentLegitCount = 0;
        self.presentSpamCount = 0;
        self.presentLegitCount = 0;

