import sys

class Configure(object):
    """Configure."""
    def __init__(self, number_of_nodes):
        self.number_of_nodes = number_of_nodes

    def installPackages(self):
        self.invoke("bash ~/disdel/config/setup")

    def deploy(self):
        self.invoke("bash ~/disdel/config/getDisdel {0}".format(self.number_of_nodes))

    def redeploy(self):
        self.terminate()
        self.deploy()

    def terminate(self):
        self.invoke("helm delete disdel ; kind delete cluster")

    def invoke(self, command):
        subprocess.run(['/bin/bash', '-c', command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


class Validator(object):
    """Validation of parameters and operations."""

    def validTask(self, task):
        return task in {'train', 'infer'}

    def validModel(self, model):
        return model in {'custom05', 'vgg16', 'resnet50', 'mobilenet'}
        # return model in {'custom05'}

    def validate(self, value, limit):
        return value < limit

class Ops(object):
    """Math operations"""

    def KBtoMB(self, size):
        return size / 1024

    def MBtoGB(self, size):
        return size / 1024

    def GBtoMB(self, size):
        return size * 1024

class Log(object):
    """Error logging."""

    class InitError():
        """Log initialization errors."""

        def __init__(self, caller):
            self.identifier = "InitError"
            if caller == "build":
                self.buildError()

        def buildError(self):
            sys.stdout.write("\u001b[31m")
            print("{0}: Unable to satisfy the request. Maximum number of actions exceeded.".format(
                self.identifier))
            sys.stdout.write("\u001b[0m")

    class ValueError():
        """Log parameter value errors."""

        def __init__(self, subject):
            self.identifier = "ValueError"
            if subject == "task":
                self.taskValue()
            elif subject == "model":
                self.modelValue()

        def taskValue(self):
            sys.stdout.write("\u001b[31m")
            print(
                "{0}: Invalid request. Valid tasks are train/infer.".format(self.identifier))
            sys.stdout.write("\u001b[0m")

        def modelValue(self):
            sys.stdout.write("\u001b[31m")
            print(
                "{0}: Invalid request. Valid models are ['custom05', 'vgg16', 'resnet50', 'mobilenet'].".format(self.identifier))
            sys.stdout.write("\u001b[0m")

    class Warning():
        """Emittion of Warnings."""

        def __init__(self):
            self.id = "Warning"

        def warnMemory(self, value, limit):
            sys.stdout.write("\u001b[33m")
            print("{0}: Request of {1}.MB exceeds the threshold of {2}.MB per action".format(
                self.id, value, limit))
            sys.stdout.write("\u001b[0m")
