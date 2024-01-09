# -*- coding: utf-8 -*-
import numpy as np


class DrivingTerms(object):
    """DrivingTerms.
            can get values by keys:
           {'h21000': , 'h30000': , 'h10110': , 'h10020': ,
            'h10200': , 'h20001': , 'h00201': , 'h10002': ,
            'h22000': , 'h11110': , 'h00220': ,
            'h31000': , 'h40000': , 'h20110': , 'h11200': ,
            'h20020': , 'h20200': , 'h00310': , 'h00400': }
        method:
            set_periods(n_periods)
                change the number of periods and the value of resonance driving terms. Return self.
            fluctuation(n_periods)
                compute the fluctuations of the RDTs along n periods (except h22000, h11110 and h00220).
                Return a dictionary of 1-dim np.ndarray.
            fluctuation_components()
                compute the components of the RDTs fluctuation.
                return a dictionary. Each value is a 2-dim np.ndarray, each row is a mode of fluctuation.
                Use sum(components[:, 0] * components[:, 1] ** k) to get the value for k periods."""

    def __init__(self, n_periods, phix, phiy,
                 f21000, f30000, f10110, f10020, f10200, f20001, f00201, f10002,
                 h22000, h11110, h00220, f31000, f40000, f20110, f11200, f20020, f20200, f00310, f00400,
                 h21000s, h30000s, h10110s, h10020s, h10200s, h20001s, h00201s, h10002s,
                 h31000s, h40000s, h20110s, h11200s, h20020s, h20200s, h00310s, h00400s):
        self.n_periods = n_periods
        self.phix = phix
        self.phiy = phiy
        self.f21000 = f21000
        self.f30000 = f30000
        self.f10110 = f10110
        self.f10020 = f10020
        self.f10200 = f10200
        self.f20001 = f20001
        self.f00201 = f00201
        self.f10002 = f10002
        self.h22000 = h22000
        self.h11110 = h11110
        self.h00220 = h00220
        self.f31000 = f31000
        self.f40000 = f40000
        self.f20110 = f20110
        self.f11200 = f11200
        self.f20020 = f20020
        self.f20200 = f20200
        self.f00310 = f00310
        self.f00400 = f00400
        self.h21000s = h21000s
        self.h30000s = h30000s
        self.h10110s = h10110s
        self.h10020s = h10020s
        self.h10200s = h10200s
        self.h20001s = h20001s
        self.h00201s = h00201s
        self.h10002s = h10002s
        self.h31000s = h31000s
        self.h40000s = h40000s
        self.h20110s = h20110s
        self.h11200s = h11200s
        self.h20020s = h20020s
        self.h20200s = h20200s
        self.h00310s = h00310s
        self.h00400s = h00400s
        self.terms = {}
        self.set_periods(n_periods)

    def set_periods(self, n_periods):
        self.n_periods = n_periods
        jj = complex(0, 1)
        q21000 = np.exp(complex(0, self.phix)) ** n_periods
        q30000 = np.exp(complex(0, self.phix * 3)) ** n_periods
        q10110 = np.exp(complex(0, self.phix)) ** n_periods
        q10020 = np.exp(complex(0, self.phix - 2 * self.phiy)) ** n_periods
        q10200 = np.exp(complex(0, self.phix + 2 * self.phiy)) ** n_periods
        q12000 = q21000.conjugate()
        q03000 = q30000.conjugate()
        q01110 = q10110.conjugate()
        q01200 = q10020.conjugate()
        q01020 = q10200.conjugate()
        q20001 = np.exp(complex(0, 2 * self.phix)) ** n_periods
        q00201 = np.exp(complex(0, 2 * self.phiy)) ** n_periods
        q10002 = np.exp(complex(0, self.phix)) ** n_periods
        # 3rd-order
        h21000 = self.f21000 * (1 - q21000)
        h30000 = self.f30000 * (1 - q30000)
        h10110 = self.f10110 * (1 - q10110)
        h10020 = self.f10020 * (1 - q10020)
        h10200 = self.f10200 * (1 - q10200)
        h20001 = self.f20001 * (1 - q20001)
        h00201 = self.f00201 * (1 - q00201)
        h10002 = self.f10002 * (1 - q10002)
        # 4th-order
        q31000 = np.exp(complex(0, 2 * self.phix)) ** n_periods
        q40000 = np.exp(complex(0, 4 * self.phix)) ** n_periods
        q20110 = np.exp(complex(0, 2 * self.phix)) ** n_periods
        q11200 = np.exp(complex(0, 2 * self.phiy)) ** n_periods
        q20020 = np.exp(complex(0, 2 * self.phix - 2 * self.phiy)) ** n_periods
        q20200 = np.exp(complex(0, 2 * self.phix + 2 * self.phiy)) ** n_periods
        q00310 = np.exp(complex(0, 2 * self.phiy)) ** n_periods
        q00400 = np.exp(complex(0, 4 * self.phiy)) ** n_periods
        h22000 = self.h22000 * n_periods
        h11110 = self.h11110 * n_periods
        h00220 = self.h00220 * n_periods
        h31000 = self.f31000 * (1 - q31000)
        h40000 = self.f40000 * (1 - q40000)
        h20110 = self.f20110 * (1 - q20110)
        h11200 = self.f11200 * (1 - q11200)
        h20020 = self.f20020 * (1 - q20020)
        h20200 = self.f20200 * (1 - q20200)
        h00310 = self.f00310 * (1 - q00310)
        h00400 = self.f00400 * (1 - q00400)
        f12000 = self.f21000.conjugate()
        f01110 = self.f10110.conjugate()
        f01200 = self.f10020.conjugate()
        f01020 = self.f10200.conjugate()

        h22000 += jj * (3 * self.f21000 * f12000 * (q21000 - q12000)
                        + (q30000 - q03000) * self.f30000.conjugate() * self.f30000 * 9)
        h11110 += jj * ((q21000 - q01110) * f01110 * self.f21000 * 2
                        - (q12000 - q10110) * self.f10110 * f12000 * 2
                        - (q10020 - q01200) * f01200 * self.f10020 * 4
                        + (q10200 - q01020) * f01020 * self.f10200 * 4)
        h00220 += jj * ((q10020 - q01200) * f01200 * self.f10020
                        + (q10200 - q01020) * f01020 * self.f10200
                        + (q10110 - q01110) * f01110 * self.f10110)
        h31000 += jj * 6 * (q30000 - q12000) * f12000 * self.f30000
        h40000 += jj * 3 * (q30000 - q21000) * self.f21000 * self.f30000
        h20110 += jj * ((q30000 - q01110) * f01110 * self.f30000 * 3
                        - (q21000 - q10110) * self.f10110 * self.f21000
                        + (q10200 - q10020) * self.f10020 * self.f10200 * 4)
        h11200 += jj * ((q10200 - q12000) * f12000 * self.f10200 * 2
                        + (q21000 - q01200) * f01200 * self.f21000 * 2
                        + (q10200 - q01110) * f01110 * self.f10200 * 2
                        + (q10110 - q01200) * f01200 * self.f10110 * (-2))
        h20020 += jj * (-(q21000 - q10020) * self.f10020 * self.f21000
                        + (q30000 - q01020) * f01020 * self.f30000 * 3
                        + (q10110 - q10020) * self.f10020 * self.f10110 * 2)
        h20200 += jj * ((q30000 - q01200) * f01200 * self.f30000 * 3
                        + (q10200 - q21000) * self.f21000 * self.f10200
                        + (q10110 - q10200) * self.f10200 * self.f10110 * (-2))
        h00310 += jj * ((q10200 - q01110) * f01110 * self.f10200
                        + (q10110 - q01200) * f01200 * self.f10110)
        h00400 += jj * (q10200 - q01200) * f01200 * self.f10200

        self.terms = {'h21000': h21000, 'h30000': h30000, 'h10110': h10110, 'h10020': h10020,
                      'h10200': h10200, 'h20001': h20001, 'h00201': h00201, 'h10002': h10002,
                      'h22000': h22000, 'h11110': h11110, 'h00220': h00220,
                      'h31000': h31000, 'h40000': h40000, 'h20110': h20110, 'h11200': h11200,
                      'h20020': h20020, 'h20200': h20200, 'h00310': h00310, 'h00400': h00400}
        return self

    def buildup_fluctuation(self, n_periods=None):
        """compute the RDTs fluctuation along n_periods periods.

        Return:
            return a dictionary. Each value is a 1-dim np.ndarray of complex number."""

        jj = complex(0, 1)
        q21000 = np.exp(complex(0, self.phix))
        q30000 = np.exp(complex(0, self.phix * 3))
        q10110 = np.exp(complex(0, self.phix))
        q10020 = np.exp(complex(0, self.phix - 2 * self.phiy))
        q10200 = np.exp(complex(0, self.phix + 2 * self.phiy))
        q12000 = q21000.conjugate()
        q01110 = q10110.conjugate()
        q01200 = q10020.conjugate()
        q01020 = q10200.conjugate()
        q20001 = np.exp(complex(0, 2 * self.phix))
        q00201 = np.exp(complex(0, 2 * self.phiy))
        q10002 = np.exp(complex(0, self.phix))

        # 4th-order
        q31000 = np.exp(complex(0, 2 * self.phix))
        q40000 = np.exp(complex(0, 4 * self.phix))
        q20110 = np.exp(complex(0, 2 * self.phix))
        q11200 = np.exp(complex(0, 2 * self.phiy))
        q20020 = np.exp(complex(0, 2 * self.phix - 2 * self.phiy))
        q20200 = np.exp(complex(0, 2 * self.phix + 2 * self.phiy))
        q00310 = np.exp(complex(0, 2 * self.phiy))
        q00400 = np.exp(complex(0, 4 * self.phiy))
        f12000 = self.f21000.conjugate()
        f01110 = self.f10110.conjugate()
        f01200 = self.f10020.conjugate()
        f01020 = self.f10200.conjugate()
        h12000s = np.conj(self.h21000s)
        h01110s = np.conj(self.h10110s)
        h01200s = np.conj(self.h10020s)
        h01020s = np.conj(self.h10200s)
        n_periods = self.n_periods if n_periods is None else n_periods

        chro_num = len(self.h20001s)
        geo_num = len(self.h21000s)
        h21000s = np.zeros(n_periods * geo_num, dtype='complex_')
        h30000s = np.zeros(n_periods * geo_num, dtype='complex_')
        h10110s = np.zeros(n_periods * geo_num, dtype='complex_')
        h10020s = np.zeros(n_periods * geo_num, dtype='complex_')
        h10200s = np.zeros(n_periods * geo_num, dtype='complex_')
        h20001s = np.zeros(n_periods * chro_num, dtype='complex_')
        h00201s = np.zeros(n_periods * chro_num, dtype='complex_')
        h10002s = np.zeros(n_periods * chro_num, dtype='complex_')
        h31000s = np.zeros(n_periods * geo_num, dtype='complex_')
        h40000s = np.zeros(n_periods * geo_num, dtype='complex_')
        h20110s = np.zeros(n_periods * geo_num, dtype='complex_')
        h11200s = np.zeros(n_periods * geo_num, dtype='complex_')
        h20020s = np.zeros(n_periods * geo_num, dtype='complex_')
        h20200s = np.zeros(n_periods * geo_num, dtype='complex_')
        h00310s = np.zeros(n_periods * geo_num, dtype='complex_')
        h00400s = np.zeros(n_periods * geo_num, dtype='complex_')
        h21000s[: geo_num] = self.h21000s
        h30000s[: geo_num] = self.h30000s
        h10110s[: geo_num] = self.h10110s
        h10020s[: geo_num] = self.h10020s
        h10200s[: geo_num] = self.h10200s
        h20001s[: chro_num] = self.h20001s
        h00201s[: chro_num] = self.h00201s
        h10002s[: chro_num] = self.h10002s
        h31000s[: geo_num] = self.h31000s
        h40000s[: geo_num] = self.h40000s
        h20110s[: geo_num] = self.h20110s
        h11200s[: geo_num] = self.h11200s
        h20020s[: geo_num] = self.h20020s
        h20200s[: geo_num] = self.h20200s
        h00310s[: geo_num] = self.h00310s
        h00400s[: geo_num] = self.h00400s
        for j in range(n_periods - 1):
            i = j + 1
            h21000s[i * geo_num: (i + 1) * geo_num] = self.f21000 + (self.h21000s - self.f21000) * q21000 ** i
            h30000s[i * geo_num: (i + 1) * geo_num] = self.f30000 + (self.h30000s - self.f30000) * q30000 ** i
            h10110s[i * geo_num: (i + 1) * geo_num] = self.f10110 + (self.h10110s - self.f10110) * q10110 ** i
            h10020s[i * geo_num: (i + 1) * geo_num] = self.f10020 + (self.h10020s - self.f10020) * q10020 ** i
            h10200s[i * geo_num: (i + 1) * geo_num] = self.f10200 + (self.h10200s - self.f10200) * q10200 ** i
            h20001s[i * chro_num: (i + 1) * chro_num] = self.f20001 + (self.h20001s - self.f20001) * q20001 ** i
            h00201s[i * chro_num: (i + 1) * chro_num] = self.f00201 + (self.h00201s - self.f00201) * q00201 ** i
            h10002s[i * chro_num: (i + 1) * chro_num] = self.f10002 + (self.h10002s - self.f10002) * q10002 ** i

            h31000s[i * geo_num: (i + 1) * geo_num] = (self.f31000 + (self.h31000s - (
                        self.f30000 * h12000s - f12000 * self.h30000s) * jj * 6 - self.f31000) * q31000 ** i
                                                       + jj * 6 * f12000 * (self.f30000 - self.h30000s) * q30000 ** i
                                                       + jj * 6 * (h12000s - f12000) * self.f30000 * q12000 ** i)
            h40000s[i * geo_num: (i + 1) * geo_num] = (self.f40000 + (self.h40000s - (
                        self.f30000 * self.h21000s - self.f21000 * self.h30000s) * jj * 3 - self.f40000) * q40000 ** i
                                                       + jj * 3 * self.f21000 * (
                                                                   self.f30000 - self.h30000s) * q30000 ** i
                                                       + jj * 3 * (
                                                                   self.h21000s - self.f21000) * self.f30000 * q21000 ** i)
            h20110s[i * geo_num: (i + 1) * geo_num] = (self.f20110 +
                                                       (self.h20110s -
                                                        ((self.f30000 * h01110s - f01110 * self.h30000s) * 3
                                                         - self.f21000 * self.h10110s + self.f10110 * self.h21000s
                                                         + (
                                                                     self.f10200 * self.h10020s - self.f10020 * self.h10200s) * 4) * jj - self.f20110) * q20110 ** i
                                                       + (f01110 * (self.f30000 - self.h30000s) * 3 * q30000 ** i
                                                          - (f01110 - h01110s) * self.f30000 * 3 * q01110 ** i
                                                          - self.f10110 * (self.f21000 - self.h21000s) * q21000 ** i
                                                          + (self.f10110 - self.h10110s) * self.f21000 * q10110 ** i
                                                          + self.f10020 * (self.f10200 - self.h10200s) * 4 * q10200 ** i
                                                          - (
                                                                      self.f10020 - self.h10020s) * self.f10200 * 4 * q10020 ** i) * jj)
            h11200s[i * geo_num: (i + 1) * geo_num] = (self.f11200 +
                                                       (self.h11200s -
                                                        (self.f10200 * (h12000s + h01110s) - f12000 * self.h10200s
                                                         + self.f21000 * h01200s - f01200 * (
                                                                     self.h21000s - self.h10110s)
                                                         - f01110 * self.h10200s - self.f10110 * h01200s) * jj * 2 - self.f11200) * q11200 ** i
                                                       + ((f12000 + f01110) * (
                                self.f10200 - self.h10200s) * 2 * q10200 ** i
                                                          - (f12000 - h12000s) * self.f10200 * 2 * q12000 ** i
                                                          + f01200 * (self.f21000 - self.h21000s) * 2 * q21000 ** i
                                                          - (f01200 - h01200s) * (
                                                                      self.f21000 - self.f10110) * 2 * q01200 ** i
                                                          - (f01110 - h01110s) * self.f10200 * 2 * q01110 ** i
                                                          + f01200 * (self.f10110 - self.h10110s) * (
                                                              -2) * q10110 ** i) * jj)
            h20020s[i * geo_num: (i + 1) * geo_num] = (self.f20020 +
                                                       (self.h20020s -
                                                        (-self.f21000 * self.h10020s + self.f10020 * (
                                                                    self.h21000s - self.h10110s * 2)
                                                         + self.f30000 * h01020s * 3 - f01020 * self.h30000s * 3
                                                         + self.f10110 * self.h10020s * 2) * jj - self.f20020) * q20020 ** i
                                                       + (-self.f10020 * (self.f21000 - self.h21000s) * q21000 ** i
                                                          + (self.f10020 - self.h10020s) * (
                                                                      self.f21000 - self.f10110 * 2) * q10020 ** i
                                                          + f01020 * (self.f30000 - self.h30000s) * 3 * q30000 ** i
                                                          - (f01020 - h01020s) * self.f30000 * 3 * q01020 ** i
                                                          + self.f10020 * (
                                                                      self.f10110 - self.h10110s) * 2 * q10110 ** i) * jj)
            h20200s[i * geo_num: (i + 1) * geo_num] = (self.f20200 +
                                                       (self.h20200s -
                                                        (self.f30000 * h01200s * 3 - f01200 * self.h30000s * 3
                                                         + self.f10200 * (self.h21000s + self.h10110s * 2)
                                                         - self.f21000 * self.h10200s
                                                         - self.f10110 * self.h10200s * 2) * jj - self.f20200) * q20200 ** i
                                                       + (f01200 * (self.f30000 - self.h30000s) * 3 * q30000 ** i
                                                          - (f01200 - h01200s) * self.f30000 * 3 * q01200 ** i
                                                          + (self.f21000 + 2 * self.f10110) * (
                                                                      self.f10200 - self.h10200s) * q10200 ** i
                                                          - (self.f21000 - self.h21000s) * self.f10200 * q21000 ** i
                                                          + self.f10200 * (self.f10110 - self.h10110s) * (
                                                              -2) * q10110 ** i) * jj)
            h00310s[i * geo_num: (i + 1) * geo_num] = (self.f00310 +
                                                       (self.h00310s -
                                                        (self.f10200 * h01110s - f01110 * self.h10200s
                                                         + self.f10110 * h01200s - f01200 * self.h10110s) * jj - self.f00310) * q00310 ** i
                                                       + (f01110 * (self.f10200 - self.h10200s) * q10200 ** i
                                                          - (f01110 - h01110s) * self.f10200 * q01110 ** i
                                                          + f01200 * (self.f10110 - self.h10110s) * q10110 ** i
                                                          - (f01200 - h01200s) * self.f10110 * q01200 ** i) * jj)
            h00400s[i * geo_num: (i + 1) * geo_num] = (self.f00400 +
                                                       (self.h00400s -
                                                        (
                                                                    self.f10200 * h01200s - f01200 * self.h10200s) * jj - self.f00400) * q00400 ** i
                                                       + (f01200 * (self.f10200 - self.h10200s) * q10200 ** i
                                                          - (f01200 - h01200s) * self.f10200 * q01200 ** i) * jj)
        return {'h21000': h21000s, 'h30000': h30000s, 'h10110': h10110s, 'h10020': h10020s,
                'h10200': h10200s, 'h20001': h20001s, 'h00201': h00201s, 'h10002': h10002s,
                'h31000': h31000s, 'h40000': h40000s, 'h20110': h20110s, 'h11200': h11200s,
                'h20020': h20020s, 'h20200': h20200s, 'h00310': h00310s, 'h00400': h00400s}

    def fluctuation_components(self):
        """compute the components of the RDTs fluctuation.

        Return:
            return a dictionary. Each value is a tuple of tuples, ((mode1, r1), (mode2, r2), ...).
        """

        jj = complex(0, 1)
        num_geo = len(self.h30000s)
        q21000 = np.exp(complex(0, self.phix))
        q30000 = np.exp(complex(0, self.phix * 3))
        q10110 = np.exp(complex(0, self.phix))  # q21000 is the same as q10110
        q10020 = np.exp(complex(0, self.phix - 2 * self.phiy))
        q10200 = np.exp(complex(0, self.phix + 2 * self.phiy))
        q12000 = q21000.conjugate()
        q01110 = q10110.conjugate()
        q01200 = q10020.conjugate()
        q01020 = q10200.conjugate()
        q20001 = np.exp(complex(0, 2 * self.phix))
        q00201 = np.exp(complex(0, 2 * self.phiy))
        q10002 = np.exp(complex(0, self.phix))

        h21000_fluct = ((1, np.zeros(num_geo) + self.f21000), (q21000, self.h21000s - self.f21000))
        h30000_fluct = ((1, np.zeros(num_geo) + self.f30000), (q30000, self.h30000s - self.f30000))
        h10110_fluct = ((1, np.zeros(num_geo) + self.f10110), (q10110, self.h10110s - self.f10110))
        h10020_fluct = ((1, np.zeros(num_geo) + self.f10020), (q10020, self.h10020s - self.f10020))
        h10200_fluct = ((1, np.zeros(num_geo) + self.f10200), (q10200, self.h10200s - self.f10200))
        h20001_fluct = ((1, np.zeros(len(self.h20001s)) + self.f20001), (q20001, self.h20001s - self.f20001))
        h00201_fluct = ((1, np.zeros(len(self.h00201s)) + self.f00201), (q00201, self.h00201s - self.f00201))
        h10002_fluct = ((1, np.zeros(len(self.h10002s)) + self.f10002), (q10002, self.h10002s - self.f10002))

        f12000 = self.f21000.conjugate()
        f01110 = self.f10110.conjugate()
        f01200 = self.f10020.conjugate()
        f01020 = self.f10200.conjugate()
        h12000s = np.conj(self.h21000s)
        h01110s = np.conj(self.h10110s)
        h01200s = np.conj(self.h10020s)
        h01020s = np.conj(self.h10200s)
        # 4th-order
        q31000 = np.exp(complex(0, 2 * self.phix))
        q40000 = np.exp(complex(0, 4 * self.phix))
        q20110 = np.exp(complex(0, 2 * self.phix))
        q11200 = np.exp(complex(0, 2 * self.phiy))
        q20020 = np.exp(complex(0, 2 * self.phix - 2 * self.phiy))
        q20200 = np.exp(complex(0, 2 * self.phix + 2 * self.phiy))
        q00310 = np.exp(complex(0, 2 * self.phiy))
        q00400 = np.exp(complex(0, 4 * self.phiy))

        h31000_fluct = ((1, self.f31000 + np.zeros(num_geo)),
                        (q31000, self.h31000s - (self.f30000 * h12000s - f12000 * self.h30000s) * jj * 6 - self.f31000),
                        (q30000, jj * 6 * f12000 * (self.f30000 - self.h30000s)),
                        (q12000, jj * 6 * (h12000s - f12000) * self.f30000))
        h40000_fluct = ((1, self.f40000 + np.zeros(num_geo)),
                        (q40000,
                         self.h40000s - (
                                     self.f30000 * self.h21000s - self.f21000 * self.h30000s) * jj * 3 - self.f40000),
                        (q30000, jj * 3 * self.f21000 * (self.f30000 - self.h30000s)),
                        (q21000, jj * 3 * (self.h21000s - self.f21000) * self.f30000))
        h20110_fluct = ((1, self.f20110 + np.zeros(num_geo)),
                        (q20110, self.h20110s - ((self.f30000 * h01110s - f01110 * self.h30000s) * 3
                                                 - self.f21000 * self.h10110s + self.f10110 * self.h21000s
                                                 + (
                                                         self.f10200 * self.h10020s - self.f10020 * self.h10200s) * 4) * jj - self.f20110),
                        (q30000, jj * f01110 * (self.f30000 - self.h30000s) * 3),
                        (q01110, -(f01110 - h01110s) * self.f30000 * 3 * jj),
                        (q21000, -self.f10110 * (self.f21000 - self.h21000s) * jj + (
                                self.f10110 - self.h10110s) * self.f21000 * jj),
                        (q10200, +self.f10020 * (self.f10200 - self.h10200s) * 4 * jj),
                        (q10020, - (self.f10020 - self.h10020s) * self.f10200 * 4 * jj))
        h11200_fluct = ((1, self.f11200 + np.zeros(num_geo)),
                        (q11200, self.h11200s - (self.f10200 * (h12000s + h01110s) - f12000 * self.h10200s
                                                 + self.f21000 * h01200s - f01200 * (self.h21000s - self.h10110s)
                                                 - f01110 * self.h10200s - self.f10110 * h01200s) * jj * 2 - self.f11200),
                        (q10200, jj * (f12000 + f01110) * (self.f10200 - self.h10200s) * 2),
                        (
                        q12000, -jj * (f12000 - h12000s) * self.f10200 * 2 - jj * (f01110 - h01110s) * self.f10200 * 2),
                        (q21000,
                         jj * f01200 * (self.f21000 - self.h21000s) * 2 - jj * 2 * f01200 * (
                                     self.f10110 - self.h10110s)),
                        (q01200, -jj * (f01200 - h01200s) * (self.f21000 - self.f10110) * 2))
        h20020_fluct = ((1, self.f20020 + np.zeros(num_geo)),
                        (q20020,
                         self.h20020s - (-self.f21000 * self.h10020s + self.f10020 * (self.h21000s - self.h10110s * 2)
                                         + self.f30000 * h01020s * 3 - f01020 * self.h30000s * 3
                                         + self.f10110 * self.h10020s * 2) * jj - self.f20020),
                        (q21000, -jj * self.f10020 * (self.f21000 - self.h21000s) + jj * self.f10020 * (
                                self.f10110 - self.h10110s) * 2),
                        (q10020, jj * (self.f10020 - self.h10020s) * (self.f21000 - self.f10110 * 2)),
                        (q30000, jj * f01020 * (self.f30000 - self.h30000s) * 3),
                        (q01020, - jj * (f01020 - h01020s) * self.f30000 * 3))
        h20200_fluct = ((1, self.f20200 + np.zeros(num_geo)),
                        (q20200, self.h20200s - (self.f30000 * h01200s * 3 - f01200 * self.h30000s * 3
                                                 + self.f10200 * (self.h21000s + self.h10110s * 2)
                                                 - self.f21000 * self.h10200s
                                                 - self.f10110 * self.h10200s * 2) * jj - self.f20200),
                        (q30000, jj * f01200 * (self.f30000 - self.h30000s) * 3),
                        (q01200, - jj * (f01200 - h01200s) * self.f30000 * 3),
                        (q10200, jj * (self.f21000 + 2 * self.f10110) * (self.f10200 - self.h10200s)),
                        (q21000, - jj * (self.f21000 - self.h21000s) * self.f10200 + jj * self.f10200 * (
                                self.f10110 - self.h10110s) * (-2)))
        h00310_fluct = ((1, self.f00310 + np.zeros(len(self.h00310s))),
                        (q00310, self.h00310s - (self.f10200 * h01110s - f01110 * self.h10200s
                                                 + self.f10110 * h01200s - f01200 * self.h10110s) * jj - self.f00310),
                        (q10200, jj * f01110 * (self.f10200 - self.h10200s)),
                        (q01110, - jj * (f01110 - h01110s) * self.f10200),
                        (q10110, jj * f01200 * (self.f10110 - self.h10110s)),
                        (q01200, - jj * (f01200 - h01200s) * self.f10110))
        h00400_fluct = ((1, self.f00400 + np.zeros(len(self.h00400s))),
                        (q00400, self.h00400s - (self.f10200 * h01200s - f01200 * self.h10200s) * jj - self.f00400),
                        (q10200, jj * f01200 * (self.f10200 - self.h10200s)),
                        (q01200, - jj * (f01200 - h01200s) * self.f10200))
        return {'h21000': h21000_fluct, 'h30000': h30000_fluct, 'h10110': h10110_fluct, 'h10020': h10020_fluct,
                'h10200': h10200_fluct, 'h20001': h20001_fluct, 'h00201': h00201_fluct, 'h10002': h10002_fluct,
                'h31000': h31000_fluct, 'h40000': h40000_fluct, 'h20110': h20110_fluct, 'h11200': h11200_fluct,
                'h20020': h20020_fluct, 'h20200': h20200_fluct, 'h00310': h00310_fluct, 'h00400': h00400_fluct}

    def natural_fluctuation(self):
        """Compute the absolute values of the natural RDT fluctuation.

        Return: A dictionary.
        {'f21000', 'f30000', 'f10110', 'f10020',
                'f10200', 'f20001', 'f00201', 'f10002',
                'f31000', 'f40000', 'f20110', 'f11200',
                'f20020', 'f20200', 'f00310', 'f00400'}
        """

        # calculate natural RDT fluctuation.
        # Here fxxxxx is f_xxxxx(0) in (Franchi,2014),
        # but fxxxxxs is not equal to f_xxxxx(s) their absolute values are equal.

        jj = complex(0, 1)

        # 4th-order
        f12000 = self.f21000.conjugate()
        f01110 = self.f10110.conjugate()
        f01200 = self.f10020.conjugate()
        f01020 = self.f10200.conjugate()
        h12000s = np.conj(self.h21000s)
        h01110s = np.conj(self.h10110s)
        h01200s = np.conj(self.h10020s)
        h01020s = np.conj(self.h10200s)

        chro_num = len(self.h20001s)
        geo_num = len(self.h21000s)
        f21000s = np.zeros(geo_num, dtype='complex_')
        f30000s = np.zeros(geo_num, dtype='complex_')
        f10110s = np.zeros(geo_num, dtype='complex_')
        f10020s = np.zeros(geo_num, dtype='complex_')
        f10200s = np.zeros(geo_num, dtype='complex_')
        f20001s = np.zeros(chro_num, dtype='complex_')
        f00201s = np.zeros(chro_num, dtype='complex_')
        f10002s = np.zeros(chro_num, dtype='complex_')
        f31000s = np.zeros(geo_num, dtype='complex_')
        f40000s = np.zeros(geo_num, dtype='complex_')
        f20110s = np.zeros(geo_num, dtype='complex_')
        f11200s = np.zeros(geo_num, dtype='complex_')
        f20020s = np.zeros(geo_num, dtype='complex_')
        f20200s = np.zeros(geo_num, dtype='complex_')
        f00310s = np.zeros(geo_num, dtype='complex_')
        f00400s = np.zeros(geo_num, dtype='complex_')
        f21000s = self.h21000s - self.f21000
        f30000s = self.h30000s - self.f30000
        f10110s = self.h10110s - self.f10110
        f10020s = self.h10020s - self.f10020
        f10200s = self.h10200s - self.f10200
        f20001s = self.h20001s - self.f20001
        f00201s = self.h00201s - self.f00201
        f10002s = self.h10002s - self.f10002
        # f1 = self.f30000 * h12000s
        # f2 = f12000 * self.h30000s
        # f31000s = (self.h31000s - (f1 - f2) * jj * 6 -self.f31000)
        f31000s = (self.h31000s - (self.f30000 * h12000s - f12000 * self.h30000s) * jj * 6 - self.f31000)

        f40000s = (self.h40000s - (self.f30000 * self.h21000s - self.f21000 * self.h30000s) * jj * 3 - self.f40000)
        f20110s = (self.h20110s - ((self.f30000 * h01110s - f01110 * self.h30000s) * 3
                                   - self.f21000 * self.h10110s + self.f10110 * self.h21000s
                                   + (self.f10200 * self.h10020s - self.f10020 * self.h10200s) * 4) * jj - self.f20110)
        f11200s = (self.h11200s - (self.f10200 * (h12000s + h01110s) - f12000 * self.h10200s
                                   + self.f21000 * h01200s - f01200 * (self.h21000s - self.h10110s)
                                   - f01110 * self.h10200s - self.f10110 * h01200s) * jj * 2 - self.f11200)
        f20020s = (self.h20020s - (-self.f21000 * self.h10020s + self.f10020 * (self.h21000s - self.h10110s * 2)
                                   + self.f30000 * h01020s * 3 - f01020 * self.h30000s * 3
                                   + self.f10110 * self.h10020s * 2) * jj - self.f20020)
        f20200s = (self.h20200s - (self.f30000 * h01200s * 3 - f01200 * self.h30000s * 3
                                   + self.f10200 * (self.h21000s + self.h10110s * 2)
                                   - self.f21000 * self.h10200s
                                   - self.f10110 * self.h10200s * 2) * jj - self.f20200)
        f00310s = (self.h00310s - (self.f10200 * h01110s - f01110 * self.h10200s
                                   + self.f10110 * h01200s - f01200 * self.h10110s) * jj - self.f00310)
        f00400s = (self.h00400s - (self.f10200 * h01200s - f01200 * self.h10200s) * jj - self.f00400)
        return {'f21000': np.abs(f21000s), 'f30000': np.abs(f30000s), 'f10110': np.abs(f10110s),
                'f10020': np.abs(f10020s),
                'f10200': np.abs(f10200s), 'f20001': np.abs(f20001s), 'f00201': np.abs(f00201s),
                'f10002': np.abs(f10002s),
                'f31000': np.abs(f31000s), 'f40000': np.abs(f40000s), 'f20110': np.abs(f20110s),
                'f11200': np.abs(f11200s),
                'f20020': np.abs(f20020s), 'f20200': np.abs(f20200s), 'f00310': np.abs(f00310s),
                'f00400': np.abs(f00400s)}

    def __getitem__(self, item):
        return self.terms[item]

    def __str__(self):
        text = f'driving terms: {self.n_periods:d} periods\n'
        for i, j in self.terms.items():
            text += f'    {str(i):7}: {abs(j):.2f}\n'
        return text
