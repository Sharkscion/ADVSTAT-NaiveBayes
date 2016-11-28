class EmailReader:
    def __init__(self, filename):
        self.filename = filename

    def read(self):
        return open(self.filename).read()