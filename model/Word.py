class Word:
    def __init__(self, content):
        self.content = content
        self.mutualInfo = 0
        self.notPresentSpamCount = 0;
        self.notPresentLegitCount = 0;
        self.presentSpamCount = 0;
        self.presentLegitCount = 0;

