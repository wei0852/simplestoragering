"""exceptions"""


class UnfinishedWork(Exception):
    """user's exception for marking the unfinished work"""

    def __init__(self, *args):
        self.name = args[0]


class ParticleLost(Exception):
    """particle lost during tracking"""

    def __init__(self, *args):
        self.name = args[0]
