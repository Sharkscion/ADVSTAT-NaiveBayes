class FileIO:
    filename = 'words.txt'
    def writeWords(self, words):
        file = open(self.filename, 'w')
        for word in words:
            file.write(word + '\n')

    def readWords(self):
        words = []
        file = open(self.filename, 'r')
        for line in file:
            words.append(line)