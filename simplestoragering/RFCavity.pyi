# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: profile=False
from .components import Element

class RFCavity(Element):
    """RFCavity(name: str = None, voltage_in_MeV: float = 0, frequency: float = 0, harmonic_number: float = 0, phase: float = 0)"""
    enable: bool
    voltage: float
    f_rf: float
    harmonic_number: float
    phase: float

    def __init__(self, name: str = None, voltage_in_MeV: float = 0, frequency: float = 0, harmonic_number: float = 0, phase: float = 0) -> None:...

