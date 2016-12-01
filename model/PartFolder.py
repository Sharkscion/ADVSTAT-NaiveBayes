class Part:
    def __init__(self, filename):
        self.filename = filename
        self.spamEmail = []
        self.legitEmail = []

    def addSpamEmail(self, email):
        self.spamEmail.append(email)

    def addLegitEmail(self, email):
        self.legitEmail.append(email)