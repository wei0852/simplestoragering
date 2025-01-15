# -*- coding: utf-8 -*-
import numpy as np


class DrivingTerms(object):
    """DrivingTerms.
            can get values by keys:
           {'h21000': , 'h30000': , 'h10110': , 'h10020': ,
            'h10200': ,
            'h22000': , 'h11110': , 'h00220': ,
            'h31000': , 'h40000': , 'h20110': , 'h11200': ,
            'h20020': , 'h20200': , 'h00310': , 'h00400': }
        method:
            set_periods(n_periods)
                change the number of periods and the value of resonance driving terms. Return self.
            buildup_fluctuation(n_periods)
                compute the build-up fluctuations of the RDTs along n periods (except h22000, h11110 and h00220).
                Return a dictionary of 1-dim np.ndarray[np.complex128].
            natural_fluctuation()
                compute the amplitude of natural RDT fluctuations along the lattice.
                Return a dictionary of 1-dim np.ndarray[np.float64].
            fluctuation_components()
                compute the components of the RDTs fluctuation.
                return a dictionary. Each value is a 2-dim np.ndarray, each row is a mode of fluctuation.
                Use sum(components[:, 0] * components[:, 1] ** k) to get the value for k periods."""

    def __init__(self, n_periods, phix, phiy,
                 f21000, f30000, f10110, f10020, f10200,
                 h22000, h11110, h00220, f31000, f40000, f20110, f11200, f20020, f20200, f00310, f00400,
                 h21000s, h30000s, h10110s, h10020s, h10200s,
                 h31000s, h40000s, h20110s, h11200s, h20020s, h20200s, h00310s, h00400s):
        self.n_periods = n_periods
        self.phix = phix
        self.phiy = phiy
        self.f21000 = f21000
        self.f30000 = f30000
        self.f10110 = f10110
        self.f10020 = f10020
        self.f10200 = f10200
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
        f12000 = self.f21000.conjugate()
        f01110 = self.f10110.conjugate()
        f01200 = self.f10020.conjugate()
        f01020 = self.f10200.conjugate()
        # 3rd-order 
        h21000 = self.f21000 * (1 - q21000)
        h30000 = self.f30000 * (1 - q30000)
        h10110 = self.f10110 * (1 - q10110)
        h10020 = self.f10020 * (1 - q10020)
        h10200 = self.f10200 * (1 - q10200)
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
                        +(q30000 - q03000) * self.f30000.conjugate() * self.f30000 * 9)
        h11110 += jj * ((q21000 - q01110) * f01110 * self.f21000 * 2
                        -(q12000 - q10110) * self.f10110 * f12000 * 2
                        -(q10020 - q01200) * f01200 * self.f10020 * 4
                        +(q10200 - q01020) * f01020 * self.f10200 * 4)
        h00220 += jj * ((q10020 - q01200) * f01200 * self.f10020
                        +(q10200 - q01020) * f01020 * self.f10200
                        +(q10110 - q01110) * f01110 * self.f10110)
        h31000 += jj * 6 * (q30000 - q12000) * f12000 * self.f30000
        h40000 += jj * 3 * (q30000 - q21000) * self.f21000 * self.f30000
        h20110 += jj * ((q30000 - q01110) * f01110 * self.f30000 * 3 
                       -(q21000 - q10110) * self.f10110 * self.f21000
                        +(q10200 - q10020) * self.f10020 * self.f10200 * 4)
        h11200 += jj * ((q10200 - q12000) * f12000 * self.f10200 * 2
                        +(q21000 - q01200) * f01200 * self.f21000 * 2
                        +(q10200 - q01110) * f01110 * self.f10200 * 2
                        +(q10110 - q01200) * f01200 * self.f10110 * (-2))
        h20020 += jj * (-(q21000 - q10020) * self.f10020 * self.f21000
                        +(q30000 - q01020) * f01020 * self.f30000 * 3
                        +(q10110 - q10020) * self.f10020 * self.f10110 * 2)
        h20200 += jj * ((q30000 - q01200) * f01200 * self.f30000 * 3
                        +(q10200 - q21000) * self.f21000 * self.f10200
                        +(q10110 - q10200) * self.f10200 * self.f10110 * (-2))
        h00310 += jj * ((q10200 - q01110) * f01110 * self.f10200
                        +(q10110 - q01200) * f01200 * self.f10110)
        h00400 += jj * (q10200 - q01200) * f01200 * self.f10200

        self.terms = {'h2100': h21000, 'h3000': h30000, 'h1011': h10110, 'h1002': h10020,
                      'h1020': h10200,
                      'h2200': h22000, 'h1111': h11110, 'h0022': h00220,
                      'h3100': h31000, 'h4000': h40000, 'h2011': h20110, 'h1120': h11200,
                      'h2002': h20020, 'h2020': h20200, 'h0031': h00310, 'h0040': h00400}
        return self

    def buildup_fluctuation(self, n_periods=None):
        """compute the RDTs fluctuation along n_periods periods.
            The starting position is fixed, and the ending position varies.

        Return:
            return a dictionary. Each value is a 1-dim np.ndarray[np.complex128]."""

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

        f12000 = self.f21000.conjugate()
        f01110 = self.f10110.conjugate()
        f01200 = self.f10020.conjugate()
        f01020 = self.f10200.conjugate()

        # 4th-order
        q31000 = np.exp(complex(0, 2 * self.phix))
        q40000 = np.exp(complex(0, 4 * self.phix))
        q20110 = np.exp(complex(0, 2 * self.phix))
        q11200 = np.exp(complex(0, 2 * self.phiy))
        q20020 = np.exp(complex(0, 2 * self.phix - 2 * self.phiy))
        q20200 = np.exp(complex(0, 2 * self.phix + 2 * self.phiy))
        q00310 = np.exp(complex(0, 2 * self.phiy))
        q00400 = np.exp(complex(0, 4 * self.phiy))

        h12000s = np.conj(self.h21000s)
        h01110s = np.conj(self.h10110s)
        h01200s = np.conj(self.h10020s)
        h01020s = np.conj(self.h10200s)
        n_periods = self.n_periods if n_periods is None else n_periods

        geo_num = len(self.h21000s)
        h21000s = np.zeros(n_periods * geo_num, dtype='complex_')
        h30000s = np.zeros(n_periods * geo_num, dtype='complex_')
        h10110s = np.zeros(n_periods * geo_num, dtype='complex_')
        h10020s = np.zeros(n_periods * geo_num, dtype='complex_')
        h10200s = np.zeros(n_periods * geo_num, dtype='complex_')
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
            
            h31000s[i * geo_num: (i + 1) * geo_num] = (self.f31000 + (self.h31000s - (self.f30000 * h12000s - f12000 * self.h30000s) * jj * 6 -self.f31000) * q31000 ** i
                                    +jj * 6 * f12000 * (self.f30000 - self.h30000s) * q30000 ** i
                                    +jj * 6 * (h12000s - f12000) * self.f30000 * q12000 ** i)
            h40000s[i * geo_num: (i + 1) * geo_num] = (self.f40000 + (self.h40000s - (self.f30000 * self.h21000s - self.f21000 * self.h30000s) * jj * 3 -self.f40000) * q40000 ** i
                                    + jj * 3 * self.f21000 * (self.f30000 - self.h30000s) * q30000 ** i
                                    +jj * 3* (self.h21000s - self.f21000) * self.f30000 * q21000 ** i)
            h20110s[i * geo_num: (i + 1) * geo_num] = (self.f20110 + 
                                                                (self.h20110s - 
                                                                ((self.f30000 * h01110s - f01110 * self.h30000s) * 3
                                                                - self.f21000 * self.h10110s + self.f10110 * self.h21000s
                                                                + (self.f10200 * self.h10020s - self.f10020 * self.h10200s) * 4) * jj -self.f20110) * q20110 ** i
                                    + (f01110 * (self.f30000 - self.h30000s) * 3 * q30000 ** i
                                    -(f01110 - h01110s) * self.f30000 * 3 * q01110 ** i
                                    -self.f10110 * (self.f21000 - self.h21000s) * q21000 ** i
                                    +(self.f10110 - self.h10110s) * self.f21000 * q10110 ** i
                                    +self.f10020 * (self.f10200 - self.h10200s) * 4 * q10200 ** i
                                    - (self.f10020 - self.h10020s) * self.f10200 * 4 * q10020 ** i) * jj)
            h11200s[i * geo_num: (i + 1) * geo_num] = (self.f11200 + 
                                                                (self.h11200s - 
                                                                (self.f10200 * (h12000s + h01110s) - f12000 * self.h10200s
                                                                +self.f21000 * h01200s - f01200 * (self.h21000s - self.h10110s)
                                                                -f01110 * self.h10200s - self.f10110 * h01200s) * jj * 2 -self.f11200) * q11200 ** i
                                    +((f12000 + f01110) * (self.f10200 - self.h10200s) * 2 * q10200 ** i
                                    -(f12000 - h12000s) * self.f10200 * 2 * q12000 ** i
                                    +f01200 * (self.f21000 - self.h21000s) * 2 * q21000 ** i
                                    -(f01200 - h01200s) * (self.f21000 - self.f10110) * 2 * q01200 ** i
                                    -(f01110 - h01110s) * self.f10200 * 2 * q01110 ** i
                                    +f01200 * (self.f10110 - self.h10110s) * (-2) * q10110 ** i) * jj)
            h20020s[i * geo_num: (i + 1) * geo_num] = (self.f20020 + 
                                                                (self.h20020s - 
                                                                (-self.f21000 * self.h10020s + self.f10020 * (self.h21000s - self.h10110s * 2)
                                                                + self.f30000 * h01020s * 3 - f01020 * self.h30000s * 3
                                                                +self.f10110 * self.h10020s * 2) * jj - self.f20020) * q20020 ** i
                                    +(-self.f10020 * (self.f21000 - self.h21000s) * q21000 ** i
                                    + (self.f10020 - self.h10020s) * (self.f21000 - self.f10110 * 2) * q10020 ** i
                                    + f01020 * (self.f30000 - self.h30000s) * 3 * q30000 ** i
                                    - (f01020 - h01020s) * self.f30000 * 3 * q01020 ** i
                                    + self.f10020 * (self.f10110 - self.h10110s) * 2 * q10110 ** i) * jj)
            h20200s[i * geo_num: (i + 1) * geo_num] = (self.f20200 +
                                                                (self.h20200s - 
                                                                (self.f30000 * h01200s * 3 - f01200 * self.h30000s * 3
                                                                + self.f10200 * (self.h21000s + self.h10110s * 2)
                                                                - self.f21000 * self.h10200s 
                                                                - self.f10110 * self.h10200s * 2) * jj - self.f20200) * q20200 ** i
                                    + (f01200 * (self.f30000 - self.h30000s) * 3 * q30000 ** i
                                    - (f01200 - h01200s) * self.f30000 * 3 * q01200 ** i
                                    + (self.f21000 + 2 * self.f10110) * (self.f10200 - self.h10200s) * q10200 ** i
                                    - (self.f21000 - self.h21000s) * self.f10200 * q21000 ** i
                                    + self.f10200 * (self.f10110 - self.h10110s) * (-2) * q10110 ** i) * jj)
            h00310s[i * geo_num: (i + 1) * geo_num] = (self.f00310 +
                                                                (self.h00310s - 
                                                                (self.f10200 * h01110s - f01110 * self.h10200s
                                                                +self.f10110 * h01200s - f01200 * self.h10110s) * jj - self.f00310) * q00310 ** i
                                    + (f01110 * (self.f10200 - self.h10200s) * q10200 ** i
                                    - (f01110 - h01110s) * self.f10200 * q01110 ** i
                                    + f01200 * (self.f10110 - self.h10110s) * q10110 ** i
                                    - (f01200 - h01200s) * self.f10110 * q01200 ** i) * jj)
            h00400s[i * geo_num: (i + 1) * geo_num] = (self.f00400 + 
                                                                (self.h00400s -
                                                                (self.f10200 * h01200s - f01200 * self.h10200s) * jj - self.f00400) * q00400 ** i
                                    + (f01200 * (self.f10200 - self.h10200s) * q10200 ** i
                                    - (f01200 - h01200s) * self.f10200 * q01200 ** i) * jj)
        return {'h2100': h21000s, 'h3000': h30000s, 'h1011': h10110s, 'h1002': h10020s,
                'h1020': h10200s,
                'h3100': h31000s, 'h4000': h40000s, 'h2011': h20110s, 'h1120': h11200s,
                'h2002': h20020s, 'h2020': h20200s, 'h0031': h00310s, 'h0040': h00400s}

    def fluctuation_components(self):
        """compute the components of the buildup RDTs fluctuation.
        
        Return:
            return a dictionary. Each value is a tuple, ((mode1, r1), (mode2, r2), ...).
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

        h21000_fluct = ((1, np.zeros(num_geo) + self.f21000), (q21000, self.h21000s - self.f21000))
        h30000_fluct = ((1, np.zeros(num_geo) + self.f30000), (q30000, self.h30000s - self.f30000))
        h10110_fluct = ((1, np.zeros(num_geo) + self.f10110), (q10110, self.h10110s - self.f10110))
        h10020_fluct = ((1, np.zeros(num_geo) + self.f10020), (q10020, self.h10020s - self.f10020))
        h10200_fluct = ((1, np.zeros(num_geo) + self.f10200), (q10200, self.h10200s - self.f10200))

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
                        (q31000, self.h31000s - (self.f30000 * h12000s - f12000 * self.h30000s) * jj * 6 -self.f31000),
                        (q30000, jj * 6 * f12000 * (self.f30000 - self.h30000s)),
                        (q12000, jj * 6 * (h12000s - f12000) * self.f30000))
        h40000_fluct = ((1, self.f40000 + np.zeros(num_geo)),
                        (q40000, self.f40000 - (self.f30000 * self.h21000s - self.f21000 * self.h30000s) * jj * 3 -self.f40000),
                        (q30000, jj * 3 * self.f21000 * (self.f30000 - self.h30000s)),
                        (q21000, jj * 3* (self.h21000s - self.f21000) * self.f30000))
        h20110_fluct = ((1, self.f20110 + np.zeros(num_geo)), 
                        (q20110, self.f20110 - ((self.f30000 * h01110s - f01110 * self.h30000s) * 3
                                              -self.f21000 * self.h10110s + self.f10110 * self.h21000s
                                              +(self.f10200 * self.h10020s - self.f10020 * self.h10200s) * 4) * jj -self.f20110),
                        (q30000, jj * f01110 * (self.f30000 - self.h30000s) * 3),
                        (q01110, -(f01110 - h01110s) * self.f30000 * 3 * jj),
                        (q21000, -self.f10110 * (self.f21000 - self.h21000s) * jj+(self.f10110 - self.h10110s) * self.f21000 * jj),
                        (q10200, +self.f10020 * (self.f10200 - self.h10200s) * 4 * jj),
                        (q10020, - (self.f10020 - self.h10020s) * self.f10200 * 4 * jj))
        h11200_fluct = ((1, self.f11200 + np.zeros(num_geo)), 
                        (q11200, self.f11200 - (self.f10200 * (h12000s + h01110s) - f12000 * self.h10200s
                                               +self.f21000 * h01200s - f01200 * (self.h21000s - self.h10110s)
                                               -f01110 * self.h10200s - self.f10110 * h01200s) * jj * 2 -self.f11200),
                        (q10200, jj * (f12000 + f01110) * (self.f10200 - self.h10200s) * 2), 
                        (q12000, -jj * (f12000 - h12000s) * self.f10200 * 2-jj * (f01110 - h01110s) * self.f10200 * 2), 
                        (q21000, jj * f01200 * (self.f21000 - self.h21000s) * 2-jj * 2 * f01200 * (self.f10110 - self.h10110s)), 
                        (q01200, -jj * (f01200 - h01200s) * (self.f21000 - self.f10110) * 2))
        h20020_fluct = ((1, self.f20020 + np.zeros(num_geo)),
                        (q20020, self.f20020 - (-self.f21000 * self.h10020s + self.f10020 * (self.h21000s - self.h10110s * 2)
                                               + self.f30000 * h01020s * 3 - f01020 * self.h30000s * 3
                                               +self.f10110 * self.h10020s * 2) * jj - self.f20020),
                        (q21000, -jj * self.f10020 * (self.f21000 - self.h21000s) + jj * self.f10020 * (self.f10110 - self.h10110s) * 2),
                        (q10020, jj * (self.f10020 - self.h10020s) * (self.f21000 - self.f10110 * 2)),
                        (q30000, jj * f01020 * (self.f30000 - self.h30000s) * 3),
                        (q01020, - jj * (f01020 - h01020s) * self.f30000 * 3))
        h20200_fluct = ((1, self.f20200 + np.zeros(num_geo)),
                        (q20200, self.f20200 - (self.f30000 * h01200s * 3 - f01200 * self.h30000s * 3
                                                + self.f10200 * (self.h21000s + self.h10110s * 2)
                                                - self.f21000 * self.h10200s 
                                                - self.f10110 * self.h10200s * 2) * jj - self.f20200),
                        (q30000, jj * f01200 * (self.f30000 - self.h30000s) * 3),
                        (q01200, - jj * (f01200 - h01200s) * self.f30000 * 3),
                        (q10200, jj * (self.f21000 + 2 * self.f10110) * (self.f10200 - self.h10200s)),
                        (q21000, - jj * (self.f21000 - self.h21000s) * self.f10200 + jj * self.f10200 * (self.f10110 - self.h10110s) * (-2)))
        h00310_fluct = ((1, self.f00310 + np.zeros(num_geo)),
                        (q00310, self.f00310 - (self.f10200 * h01110s - f01110 * self.h10200s
                                                +self.f10110 * h01200s - f01200 * self.h10110s) * jj - self.f00310),
                        (q10200, jj * f01110 * (self.f10200 - self.h10200s)),
                        (q01110, - jj * (f01110 - h01110s) * self.f10200),
                        (q10110, jj * f01200 * (self.f10110 - self.h10110s)),
                        (q01200, - jj * (f01200 - h01200s) * self.f10110))
        h00400_fluct = ((1, self.f00400 + np.zeros(num_geo)),
                        (q00400, self.f00400 - (self.f10200 * h01200s - f01200 * self.h10200s) * jj - self.f00400),
                        (q10200, jj * f01200 * (self.f10200 - self.h10200s)),
                        (q01200, - jj * (f01200 - h01200s) * self.f10200))
        return {'h2100': h21000_fluct, 'h3000': h30000_fluct, 'h1011': h10110_fluct, 'h1002': h10020_fluct,
                'h1020': h10200_fluct,
                'h3100': h31000_fluct, 'h4000': h40000_fluct, 'h2011': h20110_fluct, 'h1120': h11200_fluct,
                'h2002': h20020_fluct, 'h2020': h20200_fluct, 'h0031': h00310_fluct, 'h0040': h00400_fluct}

    def natural_fluctuation(self):
        """Compute the absolute values of the natural RDT fluctuation.
        
        Return: A dictionary. Each value is a 1-dim np.ndarray[np.float64].
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

        geo_num = len(self.h21000s)
        f21000s = np.zeros(geo_num, dtype='complex_')
        f30000s = np.zeros(geo_num, dtype='complex_')
        f10110s = np.zeros(geo_num, dtype='complex_')
        f10020s = np.zeros(geo_num, dtype='complex_')
        f10200s = np.zeros(geo_num, dtype='complex_')
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
        # f1 = self.f30000 * h12000s
        # f2 = f12000 * self.h30000s
        # f31000s = (self.h31000s - (f1 - f2) * jj * 6 -self.f31000)
        f31000s = (self.h31000s - (self.f30000 * h12000s - f12000 * self.h30000s) * jj * 6 -self.f31000)

        f40000s = (self.h40000s - (self.f30000 * self.h21000s - self.f21000 * self.h30000s) * jj * 3 -self.f40000)
        f20110s = (self.h20110s - ((self.f30000 * h01110s - f01110 * self.h30000s) * 3
                                - self.f21000 * self.h10110s + self.f10110 * self.h21000s
                                + (self.f10200 * self.h10020s - self.f10020 * self.h10200s) * 4) * jj -self.f20110)
        f11200s = (self.h11200s - (self.f10200 * (h12000s + h01110s) - f12000 * self.h10200s
                                +self.f21000 * h01200s - f01200 * (self.h21000s - self.h10110s)
                                -f01110 * self.h10200s - self.f10110 * h01200s) * jj * 2 -self.f11200)
        f20020s = (self.h20020s - (-self.f21000 * self.h10020s + self.f10020 * (self.h21000s - self.h10110s * 2)
                                + self.f30000 * h01020s * 3 - f01020 * self.h30000s * 3
                                +self.f10110 * self.h10020s * 2) * jj - self.f20020)
        f20200s = (self.h20200s - (self.f30000 * h01200s * 3 - f01200 * self.h30000s * 3
                                + self.f10200 * (self.h21000s + self.h10110s * 2)
                                - self.f21000 * self.h10200s 
                                - self.f10110 * self.h10200s * 2) * jj - self.f20200)
        f00310s = (self.h00310s - (self.f10200 * h01110s - f01110 * self.h10200s
                                +self.f10110 * h01200s - f01200 * self.h10110s) * jj - self.f00310)
        f00400s = (self.h00400s - (self.f10200 * h01200s - f01200 * self.h10200s) * jj - self.f00400)
        return {'f2100': np.abs(f21000s), 'f3000': np.abs(f30000s), 'f1011': np.abs(f10110s), 'f1002': np.abs(f10020s),
                'f1020': np.abs(f10200s),
                'f3100': np.abs(f31000s), 'f4000': np.abs(f40000s), 'f2011': np.abs(f20110s), 'f1120': np.abs(f11200s),
                'f2002': np.abs(f20020s), 'f2020': np.abs(f20200s), 'f0031': np.abs(f00310s), 'f0040': np.abs(f00400s)}

    def adts(self):
        ADTS = np.zeros(3)
        ADTS[0] = - np.real(self.terms['h2200']) * 4 / np.pi
        ADTS[1] = - np.real(self.terms['h1111']) * 2 / np.pi
        ADTS[2] = - np.real(self.terms['h0022']) * 4 / np.pi
        return ADTS

    def __getitem__(self, item):
        return self.terms[item]

    def __str__(self):
        text = f'driving terms: {self.n_periods:d} periods\n'
        for i, j in self.terms.items():
            text += f'    {str(i):7}: {abs(j):.2f}\n'
        return text


def compute_driving_terms(betax, betay, psix, psiy, k2l, k3l, period_phix, period_phiy, verbose=True):
    """compute driving terms using twiss parameters and magnet strengths.
    
    !!! Remember to use the average twiss parameters in the magnets."""

    assert len(betax) == len(betay) == len(psix) == len(psiy) == len(k2l) == len(k3l), 'data must have the same length.'
    nData = len(betax)
    jj = complex(0, 1)
    h21000s = np.zeros(nData, dtype='complex_')
    h30000s = np.zeros(nData, dtype='complex_')
    h10110s = np.zeros(nData, dtype='complex_')
    h10020s = np.zeros(nData, dtype='complex_')
    h10200s = np.zeros(nData, dtype='complex_')
    h22000s = np.zeros(nData, dtype='complex_')
    h11110s = np.zeros(nData, dtype='complex_')
    h00220s = np.zeros(nData, dtype='complex_')
    h31000s = np.zeros(nData, dtype='complex_')
    h40000s = np.zeros(nData, dtype='complex_')
    h20110s = np.zeros(nData, dtype='complex_')
    h11200s = np.zeros(nData, dtype='complex_')
    h20020s = np.zeros(nData, dtype='complex_')
    h20200s = np.zeros(nData, dtype='complex_')
    h00310s = np.zeros(nData, dtype='complex_')
    h00400s = np.zeros(nData, dtype='complex_')

    h21000 = h30000 = h10110 = h10020 = h10200 = h22000 = h11110 = h00220 = h31000 = h40000 = h20110 = h11200 = h20020 = h20200 = h00310 = h00400 = 0
    for i in range(nData):
        betax_i = betax[i]
        betay_i = betay[i]
        phix_i = psix[i]
        phiy_i = psiy[i]
        if k2l[i] != 0:
            b3l = k2l[i] / 2
            h21000j = -b3l * betax_i ** 1.5 * np.exp(jj * phix_i) / 8
            h30000j = -b3l * betax_i ** 1.5 * np.exp(jj * 3 * phix_i) / 24
            h10110j = b3l * betax_i ** 0.5 * betay_i * np.exp(jj * phix_i) / 4
            h10020j = b3l * betax_i ** 0.5 * betay_i * np.exp(jj * (phix_i - 2 * phiy_i)) / 8
            h10200j = b3l * betax_i ** 0.5 * betay_i * np.exp(jj * (phix_i + 2 * phiy_i)) / 8
    
            h12000j = h21000j.conjugate()
            h01110j = h10110j.conjugate()
            h01200j = h10020j.conjugate()
    
            h12000 = h21000.conjugate()
            h01110 = h10110.conjugate()
            h01200 = h10020.conjugate()
    
            h22000 = h22000 + jj * ((h21000 * h12000j - h12000 * h21000j) * 3 
                      + (h30000 * h30000j.conjugate() - h30000.conjugate() * h30000j) * 9)
    
            h11110 = h11110 + jj * ((h21000 * h01110j - h01110 * h21000j) * 2 
                      - (h12000 * h10110j - h10110 * h12000j) * 2 
                      - (h10020 * h01200j - h01200 * h10020j) * 4 
                      + (h10200 * h10200j.conjugate() - h10200.conjugate() * h10200j) * 4)
    
            h00220 = h00220 + jj * ((h10020 * h01200j - h01200 * h10020j) 
                      + (h10200 * h10200j.conjugate() - h10200.conjugate() * h10200j) 
                      + (h10110 * h01110j - h01110 * h10110j))
    
            h31000 = h31000 + jj * (h30000 * h12000j - h12000 * h30000j) * 6
    
            h40000 = h40000 + jj * (h30000 * h21000j - h21000 * h30000j) * 3
    
            h20110 = h20110 + jj * ((h30000 * h01110j - h01110 * h30000j) * 3 
                      - (h21000 * h10110j - h10110 * h21000j) 
                      + (h10200 * h10020j - h10020 * h10200j) * 4)
    
            h11200 = h11200 + jj * ((h10200 * h12000j - h12000 * h10200j) * 2 
                      + (h21000 * h01200j - h01200 * h21000j) * 2 
                      + (h10200 * h01110j - h01110 * h10200j) * 2 
                      - (h10110 * h01200j - h01200 * h10110j) * 2)
    
            h20020 = h20020 + jj * (-(h21000 * h10020j - h10020 * h21000j) 
                      + (h30000 * h10200j.conjugate() - h10200.conjugate() * h30000j) * 3 
                      + (h10110 * h10020j - h10020 * h10110j) * 2)
    
            h20200 = h20200 + jj * ((h30000 * h01200j - h01200 * h30000j) * 3 
                      + (h10200 * h21000j - h21000 * h10200j) 
                      - (h10110 * h10200j - h10200 * h10110j) * 2)
    
            h00310 = h00310 + jj * ((h10200 * h01110j - h01110 * h10200j) 
                      + (h10110 * h01200j - h01200 * h10110j))
    
            h00400 = h00400 + jj * (h10200 * h01200j - h01200 * h10200j)
    
            h21000 = h21000 + h21000j
            h30000 = h30000 + h30000j
            h10110 = h10110 + h10110j
            h10020 = h10020 + h10020j
            h10200 = h10200 + h10200j
        if k3l[i] != 0:
            b4l = k3l[i] / 6
            h22000 = h22000 - 3 * b4l * betax_i ** 2 / 32
            h11110 = h11110 + 3 * b4l * betax_i * betay_i / 8
            h00220 = h00220 - 3 * b4l * betay_i ** 2 / 32
            
            h31000 = h31000 - b4l * betax_i ** 2 * np.exp(jj * 2 * phix_i) / 16
            h40000 = h40000 - b4l * betax_i ** 2 * np.exp(jj * 4 * phix_i) / 64
            h20110 = h20110 + 3 * b4l * betax_i * betay_i * np.exp(jj * 2 * phix_i) / 16
            h11200 = h11200 + 3 * b4l * betax_i * betay_i * np.exp(jj * 2 * phiy_i) / 16
            h20020 = h20020 + 3 * b4l * betax_i * betay_i * np.exp(jj * (2 * phix_i - 2 * phiy_i)) / 32
            h20200 = h20200 + 3 * b4l * betax_i * betay_i * np.exp(jj * (2 * phix_i + 2 * phiy_i)) / 32
            h00310 = h00310 - b4l * betay_i ** 2 * np.exp(jj * 2 * phiy_i) / 16
            h00400 = h00400 - b4l * betay_i ** 2 * np.exp(jj * 4 * phiy_i) / 64
        h21000s[i] = h21000
        h30000s[i] = h30000
        h10110s[i] = h10110
        h10020s[i] = h10020
        h10200s[i] = h10200
        h22000s[i] = h22000
        h11110s[i] = h11110
        h00220s[i] = h00220
        h31000s[i] = h31000
        h40000s[i] = h40000
        h20110s[i] = h20110
        h11200s[i] = h11200
        h20020s[i] = h20020
        h20200s[i] = h20200
        h00310s[i] = h00310
        h00400s[i] = h00400

    # return {'h2100': h21000s, 'h3000': h30000s, 'h1011': h10110s, 'h1002': h10020s,
    #         'h1020': h10200s,
    #         'h3100': h31000s, 'h4000': h40000s, 'h2011': h20110s, 'h1120': h11200s,
    #         'h2002': h20020s, 'h2020': h20200s, 'h0031': h00310s, 'h0040': h00400s}

    phix = period_phix
    phiy = period_phiy
    f21000 = h21000 / (1 - np.exp((phix) * jj))
    f30000 = h30000 / (1 - np.exp((phix * 3) * jj))
    f10110 = h10110 / (1 - np.exp((phix) * jj))
    f10020 = h10020 / (1 - np.exp((phix - 2 * phiy) * jj))
    f10200 = h10200 / (1 - np.exp((phix + 2 * phiy) * jj))
    f12000 = f21000.conjugate()
    f01110 = f10110.conjugate()
    f01200 = f10020.conjugate()
    f01020 = f10200.conjugate()
    h12000 = h21000.conjugate()
    h01110 = h10110.conjugate()
    h01200 = h10020.conjugate()
    h01020 = h10200.conjugate()
    h22000 = jj * ((h21000 * f12000 - h12000 * f21000) * 3
                    +(h30000 * f30000.conjugate() - h30000.conjugate() * f30000) * 9) + h22000
    h11110 = jj * ((h21000 * f01110 - h01110 * f21000) * 2
                    -(h12000 * f10110 - h10110 * f12000) * 2
                    -(h10020 * f01200 - h01200 * f10020) * 4
                    +(h10200 * f01020 - h01020 * f10200) * 4) + h11110
    h00220 = jj * ((h10020 * f01200 - h01200 * f10020)
                    +(h10200 * f01020 - h01020 * f10200)
                    +(h10110 * f01110 - h01110 * f10110)) + h00220
    h31000 = jj * 6 * (h30000 * f12000 - h12000 * f30000) + h31000
    h40000 = jj * 3 * (h30000 * f21000 - h21000 * f30000) + h40000
    h20110 = jj * ((h30000 * f01110 - h01110 * f30000) * 3 
                   -(h21000 * f10110 - h10110 * f21000)
                    +(h10200 * f10020 - h10020 * f10200) * 4) + h20110
    h11200 = jj * ((h10200 * f12000 - h12000 * f10200) * 2
                    +(h21000 * f01200 - h01200 * f21000) * 2
                    +(h10200 * f01110 - h01110 * f10200) * 2
                    +(h10110 * f01200 - h01200 * f10110) * (-2)) + h11200
    h20020 = jj * (-(h21000 * f10020 - h10020 * f21000)
                    +(h30000 * f01020 - h01020 * f30000) * 3
                    +(h10110 * f10020 - h10020 * f10110) * 2) + h20020
    h20200 = jj * ((h30000 * f01200 - h01200 * f30000) * 3
                    +(h10200 * f21000 - h21000 * f10200)
                    +(h10110 * f10200 - h10200 * f10110) * (-2)) + h20200
    h00310 = jj * ((h10200 * f01110 - h01110 * f10200)
                    +(h10110 * f01200 - h01200 * f10110)) + h00310
    h00400 = jj * (h10200 * f01200 - h01200 * f10200) + h00400
    f31000 = h31000 / (1 - np.exp((2 * phix) * jj))
    f40000 = h40000 / (1 - np.exp((4 * phix) * jj))
    f20110 = h20110 / (1 - np.exp((2 * phix) * jj))
    f11200 = h11200 / (1 - np.exp((2 * phiy) * jj))
    f20020 = h20020 / (1 - np.exp((2 * phix - 2 * phiy) * jj))
    f20200 = h20200 / (1 - np.exp((2 * phix + 2 * phiy) * jj))
    f00310 = h00310 / (1 - np.exp((2 * phiy) * jj ))
    f00400 = h00400 / (1 - np.exp((4 * phiy) * jj ))
    driving_terms = DrivingTerms(1, phix, phiy,
            f21000, f30000, f10110, f10020, f10200,
            h22000, h11110, h00220, f31000, f40000, f20110, f11200, f20020, f20200, f00310, f00400, 
            h21000s, h30000s, h10110s, h10020s,
            h10200s,
            h31000s, h40000s, h20110s, h11200s,
            h20020s, h20200s, h00310s, h00400s)
    if verbose:
        print(driving_terms)
    return driving_terms