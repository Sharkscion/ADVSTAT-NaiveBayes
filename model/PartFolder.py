class PartFolder:
    def __init__(self):
        self.spamEmail = []
        self.legitEmail = []

    def addSpamEmail(self, email):
        self.spamEmail.append(email)

    def addLegitEmail(self, email):
        self.legitEmail.append(email)