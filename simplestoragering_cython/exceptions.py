# -*- coding: utf-8 -*-
class UnfinishedWork(Exception):
    """user's exception for marking the unfinished work"""

    def __init__(self, *args):
        self.name = args[0]


class ParticleLost(Exception):
    """particle lost during tracking"""

    def __init__(self, location, *args):
        self.location = location
        self.otherinf = args

    def __str__(self):
        return f'particle lost at s = {self.location} m.'
