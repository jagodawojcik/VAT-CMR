import subprocess

class Logger:
    def __init__(self, use_echo=False):
        self.use_echo = use_echo

    def log(self, message):
        if self.use_echo:
            subprocess.run(['echo', message])
        else:
            print(message)

logger = Logger()

