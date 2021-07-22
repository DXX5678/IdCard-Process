class MyError(Exception):
    def __init__(self, value):
        self.value = value