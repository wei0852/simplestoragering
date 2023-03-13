# -*- coding: utf-8 -*-
import numpy as np


class NonlinearTerms(object):
    """NonlinearTerms.
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
                 R21000, R30000, R10110, R10020, R10200, R20001, R00201, R10002,
                 h22000, h11110, h00220, R31000, R40000, R20110, R11200, R20020, R20200, R00310, R00400,
                 f21000, f30000, f10110, f10020, f10200, f20001, f00201, f10002,
                 f31000, f40000, f20110, f11200, f20020, f20200, f00310, f00400):
        self.n_periods = n_periods
        self.phix = phix
        self.phiy = phiy
        self.R21000 = R21000
        self.R30000 = R30000
        self.R10110 = R10110
        self.R10020 = R10020
        self.R10200 = R10200
        self.R20001 = R20001
        self.R00201 = R00201
        self.R10002 = R10002
        self.h22000 = h22000
        self.h11110 = h11110
        self.h00220 = h00220
        self.R31000 = R31000
        self.R40000 = R40000
        self.R20110 = R20110
        self.R11200 = R11200
        self.R20020 = R20020
        self.R20200 = R20200
        self.R00310 = R00310
        self.R00400 = R00400
        self.f21000 = f21000[1:]
        self.f30000 = f30000[1:]
        self.f10110 = f10110[1:]
        self.f10020 = f10020[1:]
        self.f10200 = f10200[1:]
        self.f20001 = f20001[1:]
        self.f00201 = f00201[1:]
        self.f10002 = f10002[1:]
        self.f31000 = f31000[1:]
        self.f40000 = f40000[1:]
        self.f20110 = f20110[1:]
        self.f11200 = f11200[1:]
        self.f20020 = f20020[1:]
        self.f20200 = f20200[1:]
        self.f00310 = f00310[1:]
        self.f00400 = f00400[1:]
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
        h21000 = self.R21000 * (1 - q21000)
        h30000 = self.R30000 * (1 - q30000)
        h10110 = self.R10110 * (1 - q10110)
        h10020 = self.R10020 * (1 - q10020)
        h10200 = self.R10200 * (1 - q10200)
        h20001 = self.R20001 * (1 - q20001)
        h00201 = self.R00201 * (1 - q00201)
        h10002 = self.R10002 * (1 - q10002)
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
        h31000 = self.R31000 * (1 - q31000)
        h40000 = self.R40000 * (1 - q40000)
        h20110 = self.R20110 * (1 - q20110)
        h11200 = self.R11200 * (1 - q11200)
        h20020 = self.R20020 * (1 - q20020)
        h20200 = self.R20200 * (1 - q20200)
        h00310 = self.R00310 * (1 - q00310)
        h00400 = self.R00400 * (1 - q00400)
        R12000 = self.R21000.conjugate()
        R01110 = self.R10110.conjugate()
        R01200 = self.R10020.conjugate()
        R01020 = self.R10200.conjugate()

        h22000 += jj * (3 * self.R21000 * R12000 * (q21000 - q12000)
                        +(q30000 - q03000) * self.R30000.conjugate() * self.R30000 * 9)
        h11110 += jj * ((q21000 - q01110) * R01110 * self.R21000 * 2
                        -(q12000 - q10110) * self.R10110 * R12000 * 2
                        -(q10020 - q01200) * R01200 * self.R10020 * 4
                        +(q10200 - q01020) * R01020 * self.R10200 * 4)
        h00220 += jj * ((q10020 - q01200) * R01200 * self.R10020
                        +(q10200 - q01020) * R01020 * self.R10200
                        +(q10110 - q01110) * R01110 * self.R10110)
        h31000 += jj * 6 * (q30000 - q12000) * R12000 * self.R30000
        h40000 += jj * 3 * (q30000 - q21000) * self.R21000 * self.R30000
        h20110 += jj * ((q30000 - q01110) * R01110 * self.R30000 * 3 
                       -(q21000 - q10110) * self.R10110 * self.R21000
                        +(q10200 - q10020) * self.R10020 * self.R10200 * 4)
        h11200 += jj * ((q10200 - q12000) * R12000 * self.R10200 * 2
                        +(q21000 - q01200) * R01200 * self.R21000 * 2
                        +(q10200 - q01110) * R01110 * self.R10200 * 2
                        +(q10110 - q01200) * R01200 * self.R10110 * (-2))
        h20020 += jj * (-(q21000 - q10020) * self.R10020 * self.R21000
                        +(q30000 - q01020) * R01020 * self.R30000 * 3
                        +(q10110 - q10020) * self.R10020 * self.R10110 * 2)
        h20200 += jj * ((q30000 - q01200) * R01200 * self.R30000 * 3
                        +(q10200 - q21000) * self.R21000 * self.R10200
                        +(q10110 - q10200) * self.R10200 * self.R10110 * (-2))
        h00310 += jj * ((q10200 - q01110) * R01110 * self.R10200
                        +(q10110 - q01200) * R01200 * self.R10110)
        h00400 += jj * (q10200 - q01200) * R01200 * self.R10200

        self.terms = {'h21000': h21000, 'h30000': h30000, 'h10110': h10110, 'h10020': h10020,
                      'h10200': h10200, 'h20001': h20001, 'h00201': h00201, 'h10002': h10002,
                      'h22000': h22000, 'h11110': h11110, 'h00220': h00220,
                      'h31000': h31000, 'h40000': h40000, 'h20110': h20110, 'h11200': h11200,
                      'h20020': h20020, 'h20200': h20200, 'h00310': h00310, 'h00400': h00400}
        return self

    def fluctuation(self, n_periods=None):
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
        R12000 = self.R21000.conjugate()
        R01110 = self.R10110.conjugate()
        R01200 = self.R10020.conjugate()
        R01020 = self.R10200.conjugate()
        f12000 = np.conj(self.f21000)
        f01110 = np.conj(self.f10110)
        f01200 = np.conj(self.f10020)
        f01020 = np.conj(self.f10200)
        n_periods = self.n_periods if n_periods is None else n_periods

        chro_num = len(self.f20001)
        geo_num = len(self.f21000)
        f21000 = np.zeros(n_periods * geo_num + 1, dtype='complex_')
        f30000 = np.zeros(n_periods * geo_num + 1, dtype='complex_')
        f10110 = np.zeros(n_periods * geo_num + 1, dtype='complex_')
        f10020 = np.zeros(n_periods * geo_num + 1, dtype='complex_')
        f10200 = np.zeros(n_periods * geo_num + 1, dtype='complex_')
        f20001 = np.zeros(n_periods * chro_num + 1, dtype='complex_')
        f00201 = np.zeros(n_periods * chro_num + 1, dtype='complex_')
        f10002 = np.zeros(n_periods * chro_num + 1, dtype='complex_')
        f31000 = np.zeros(n_periods * geo_num + 1, dtype='complex_')
        f40000 = np.zeros(n_periods * geo_num + 1, dtype='complex_')
        f20110 = np.zeros(n_periods * geo_num + 1, dtype='complex_')
        f11200 = np.zeros(n_periods * geo_num + 1, dtype='complex_')
        f20020 = np.zeros(n_periods * geo_num + 1, dtype='complex_')
        f20200 = np.zeros(n_periods * geo_num + 1, dtype='complex_')
        f00310 = np.zeros(n_periods * geo_num + 1, dtype='complex_')
        f00400 = np.zeros(n_periods * geo_num + 1, dtype='complex_')
        f21000[1: geo_num + 1] = self.f21000
        f30000[1: geo_num + 1] = self.f30000
        f10110[1: geo_num + 1] = self.f10110
        f10020[1: geo_num + 1] = self.f10020
        f10200[1: geo_num + 1] = self.f10200
        f20001[1: chro_num + 1] = self.f20001
        f00201[1: chro_num + 1] = self.f00201
        f10002[1: chro_num + 1] = self.f10002
        f31000[1: geo_num + 1] = self.f31000
        f40000[1: geo_num + 1] = self.f40000
        f20110[1: geo_num + 1] = self.f20110
        f11200[1: geo_num + 1] = self.f11200
        f20020[1: geo_num + 1] = self.f20020
        f20200[1: geo_num + 1] = self.f20200
        f00310[1: geo_num + 1] = self.f00310
        f00400[1: geo_num + 1] = self.f00400
        for j in range(n_periods - 1):
            i = j + 1
            f21000[i * geo_num + 1: (i + 1) * geo_num + 1] = self.R21000 + (self.f21000 - self.R21000) * q21000 ** i
            f30000[i * geo_num + 1: (i + 1) * geo_num + 1] = self.R30000 + (self.f30000 - self.R30000) * q30000 ** i
            f10110[i * geo_num + 1: (i + 1) * geo_num + 1] = self.R10110 + (self.f10110 - self.R10110) * q10110 ** i
            f10020[i * geo_num + 1: (i + 1) * geo_num + 1] = self.R10020 + (self.f10020 - self.R10020) * q10020 ** i
            f10200[i * geo_num + 1: (i + 1) * geo_num + 1] = self.R10200 + (self.f10200 - self.R10200) * q10200 ** i
            f20001[i * chro_num + 1: (i + 1) * chro_num + 1] = self.R20001 + (self.f20001 - self.R20001) * q20001 ** i
            f00201[i * chro_num + 1: (i + 1) * chro_num + 1] = self.R00201 + (self.f00201 - self.R00201) * q00201 ** i
            f10002[i * chro_num + 1: (i + 1) * chro_num + 1] = self.R10002 + (self.f10002 - self.R10002) * q10002 ** i
            
            f31000[i * geo_num + 1: (i + 1) * geo_num + 1] = (self.R31000 + (self.f31000 - (self.R30000 * f12000 - R12000 * self.f30000) * jj * 6 -self.R31000) * q31000 ** i
                                    +jj * 6 * R12000 * (self.R30000 - self.f30000) * q30000 ** i
                                    +jj * 6 * (f12000 - R12000) * self.R30000 * q12000 ** i)
            f40000[i * geo_num + 1: (i + 1) * geo_num + 1] = (self.R40000 + (self.f40000 - (self.R30000 * self.f21000 - self.R21000 * self.f30000) * jj * 3 -self.R40000) * q40000 ** i
                                    + jj * 3 * self.R21000 * (self.R30000 - self.f30000) * q30000 ** i
                                    +jj * 3* (self.f21000 - self.R21000) * self.R30000 * q21000 ** i)
            f20110[i * geo_num + 1: (i + 1) * geo_num + 1] = (self.R20110 + 
                                                                (self.f20110 - 
                                                                ((self.R30000 * f01110 - R01110 * self.f30000) * 3
                                                                - self.R21000 * self.f10110 + self.R10110 * self.f21000
                                                                + (self.R10200 * self.f10020 - self.R10020 * self.f10200) * 4) * jj -self.R20110) * q20110 ** i
                                    + (R01110 * (self.R30000 - self.f30000) * 3 * q30000 ** i
                                    -(R01110 - f01110) * self.R30000 * 3 * q01110 ** i
                                    -self.R10110 * (self.R21000 - self.f21000) * q21000 ** i
                                    +(self.R10110 - self.f10110) * self.R21000 * q10110 ** i
                                    +self.R10020 * (self.R10200 - self.f10200) * 4 * q10200 ** i
                                    - (self.R10020 - self.f10020) * self.R10200 * 4 * q10020 ** i) * jj)
            f11200[i * geo_num + 1: (i + 1) * geo_num + 1] = (self.R11200 + 
                                                                (self.f11200 - 
                                                                (self.R10200 * (f12000 + f01110) - R12000 * self.f10200
                                                                +self.R21000 * f01200 - R01200 * (self.f21000 - self.f10110)
                                                                -R01110 * self.f10200 - self.R10110 * f01200) * jj * 2 -self.R11200) * q11200 ** i
                                    +((R12000 + R01110) * (self.R10200 - self.f10200) * 2 * q10200 ** i
                                    -(R12000 - f12000) * self.R10200 * 2 * q12000 ** i
                                    +R01200 * (self.R21000 - self.f21000) * 2 * q21000 ** i
                                    -(R01200 - f01200) * (self.R21000 - self.R10110) * 2 * q01200 ** i
                                    -(R01110 - f01110) * self.R10200 * 2 * q01110 ** i
                                    +R01200 * (self.R10110 - self.f10110) * (-2) * q10110 ** i) * jj)
            f20020[i * geo_num + 1: (i + 1) * geo_num + 1] = (self.R20020 + 
                                                                (self.f20020 - 
                                                                (-self.R21000 * self.f10020 + self.R10020 * (self.f21000 - self.f10110 * 2)
                                                                + self.R30000 * f01020 * 3 - R01020 * self.f30000 * 3
                                                                +self.R10110 * self.f10020 * 2) * jj - self.R20020) * q20020 ** i
                                    +(-self.R10020 * (self.R21000 - self.f21000) * q21000 ** i
                                    + (self.R10020 - self.f10020) * (self.R21000 - self.R10110 * 2) * q10020 ** i
                                    + R01020 * (self.R30000 - self.f30000) * 3 * q30000 ** i
                                    - (R01020 - f01020) * self.R30000 * 3 * q01020 ** i
                                    + self.R10020 * (self.R10110 - self.f10110) * 2 * q10110 ** i) * jj)
            f20200[i * geo_num + 1: (i + 1) * geo_num + 1] = (self.R20200 +
                                                                (self.f20200 - 
                                                                (self.R30000 * f01200 * 3 - R01200 * self.f30000 * 3
                                                                + self.R10200 * (self.f21000 + self.f10110 * 2)
                                                                - self.R21000 * self.f10200 
                                                                - self.R10110 * self.f10200 * 2) * jj - self.R20200) * q20200 ** i
                                    + (R01200 * (self.R30000 - self.f30000) * 3 * q30000 ** i
                                    - (R01200 - f01200) * self.R30000 * 3 * q01200 ** i
                                    + (self.R21000 + 2 * self.R10110) * (self.R10200 - self.f10200) * q10200 ** i
                                    - (self.R21000 - self.f21000) * self.R10200 * q21000 ** i
                                    + self.R10200 * (self.R10110 - self.f10110) * (-2) * q10110 ** i) * jj)
            f00310[i * geo_num + 1: (i + 1) * geo_num + 1] = (self.R00310 +
                                                                (self.f00310 - 
                                                                (self.R10200 * f01110 - R01110 * self.f10200
                                                                +self.R10110 * f01200 - R01200 * self.f10110) * jj - self.R00310) * q00310 ** i
                                    + (R01110 * (self.R10200 - self.f10200) * q10200 ** i
                                    - (R01110 - f01110) * self.R10200 * q01110 ** i
                                    + R01200 * (self.R10110 - self.f10110) * q10110 ** i
                                    - (R01200 - f01200) * self.R10110 * q01200 ** i) * jj)
            f00400[i * geo_num + 1: (i + 1) * geo_num + 1] = (self.R00400 + 
                                                                (self.f00400 -
                                                                (self.R10200 * f01200 - R01200 * self.f10200) * jj - self.R00400) * q00400 ** i
                                    + (R01200 * (self.R10200 - self.f10200) * q10200 ** i
                                    - (R01200 - f01200) * self.R10200 * q01200 ** i) * jj)
        return {'h21000': f21000, 'h30000': f30000, 'h10110': f10110, 'h10020': f10020,
                'h10200': f10200, 'h20001': f20001, 'h00201': f00201, 'h10002': f10002,
                'h31000': f31000, 'h40000': f40000, 'h20110': f20110, 'h11200': f11200,
                'h20020': f20020, 'h20200': f20200, 'h00310': f00310, 'h00400': f00400}

    def fluctuation_components(self):
        """compute the components of the RDTs fluctuation.
        
        Return:
            return a dictionary. Each value is a 2-dim np.ndarray, each row is a mode of fluctuation.
            Use sum(components[:, 0] * components[:, 1] ** k) to get the value for k periods."""

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

        h21000_fluct = np.array([[self.R21000, 1], [-self.R21000, q21000]])
        h30000_fluct = np.array([[self.R30000, 1], [-self.R30000, q30000]])
        h10110_fluct = np.array([[self.R10110, 1], [-self.R10110, q10110]])
        h10020_fluct = np.array([[self.R10020, 1], [-self.R10020, q10020]])
        h10200_fluct = np.array([[self.R10200, 1], [-self.R10200, q10200]])
        h20001_fluct = np.array([[self.R20001, 1], [-self.R20001, q20001]])
        h00201_fluct = np.array([[self.R00201, 1], [-self.R00201, q00201]])
        h10002_fluct = np.array([[self.R10002, 1], [-self.R10002, q10002]])

        # 4th-order
        q31000 = np.exp(complex(0, 2 * self.phix))
        q40000 = np.exp(complex(0, 4 * self.phix))
        q20110 = np.exp(complex(0, 2 * self.phix))
        q11200 = np.exp(complex(0, 2 * self.phiy))
        q20020 = np.exp(complex(0, 2 * self.phix - 2 * self.phiy))
        q20200 = np.exp(complex(0, 2 * self.phix + 2 * self.phiy))
        q00310 = np.exp(complex(0, 2 * self.phiy))
        q00400 = np.exp(complex(0, 4 * self.phiy))
        R12000 = self.R21000.conjugate()
        R01110 = self.R10110.conjugate()
        R01200 = self.R10020.conjugate()
        R01020 = self.R10200.conjugate()
        h31000_fluct = np.array([[self.R31000, 1],
                                [-self.R31000, q31000],
                                [jj * 6 * R12000 * self.R30000, q30000],
                                [-jj * 6 * R12000 * self.R30000, q12000]])
        h40000_fluct = np.array([[self.R40000, 1],
                                [-self.R40000, q40000],
                                [jj * 3 * self.R21000 * self.R30000, q30000],
                                [-jj * 3* self.R21000 * self.R30000, q21000]])
        h20110_fluct = np.array([[self.R20110, 1], 
                                [-self.R20110, q20110],
                                [jj * R01110 * self.R30000 * 3, q30000],
                                [-jj * R01110 * self.R30000 * 3, q01110],
                                [-jj * self.R10110 * self.R21000, q21000],
                                [jj * self.R10110 * self.R21000, q10110],
                                [jj * self.R10020 * self.R10200 * 4, q10200],
                                [- jj * self.R10020 * self.R10200 * 4, q10020]])
        h11200_fluct = np.array([[self.R11200, 1], 
                                [-self.R11200, q11200],
                                [jj * (R12000 + R01110) * self.R10200 * 2, q10200],
                                [-jj * R12000 * self.R10200 * 2, q12000],
                                [jj * R01200 * self.R21000 * 2, q21000],
                                [-jj * R01200 * (self.R21000 - self.R10110) * 2, q01200],
                                [-jj * R01110 * self.R10200 * 2, q01110],
                                [jj * R01200 * self.R10110 * (-2), q10110]])
        h20020_fluct = np.array([[self.R20020, 1], 
                                [-self.R20020, q20020],
                                [- jj  * self.R10020 * self.R21000, q21000],
                                [ jj  * self.R10020 * (self.R21000 - self.R10110 * 2), q10020],
                                [  jj * R01020 * self.R30000 * 3, q30000],
                                [ - jj * R01020 * self.R30000 * 3, q01020],
                                [jj * self.R10020 * self.R10110 * 2, q10110]])
        h20200_fluct = np.array([[self.R20200, 1], 
                                [-self.R20200, q20200],
                                [jj * R01200 * self.R30000 * 3, q30000],
                                [ - jj * R01200 * self.R30000 * 3, q01200],
                                [jj * (self.R21000 + 2 * self.R10110) * self.R10200, q10200],
                                [ - jj * self.R21000 * self.R10200, q21000],
                                [jj * self.R10200 * self.R10110 * (-2), q10110]])
        h00310_fluct = np.array([[self.R00310, 1], 
                                [-self.R00310, q00310],
                                [jj * R01110 * self.R10200, q10200],
                                [ - jj * R01110 * self.R10200, q01110],
                                [jj * R01200 * self.R10110, q10110],
                                [ - jj * R01200 * self.R10110, q01200]])
        h00400_fluct = np.array([[self.R00400, 1], 
                                [-self.R00400, q00400],
                                [jj * R01200 * self.R10200, q10200],
                                [-jj * R01200 * self.R10200, q01200]])
        return {'h21000': h21000_fluct, 'h30000': h30000_fluct, 'h10110': h10110_fluct, 'h10020': h10020_fluct,
                'h10200': h10200_fluct, 'h20001': h20001_fluct, 'h00201': h00201_fluct, 'h10002': h10002_fluct,
                'h31000': h31000_fluct, 'h40000': h40000_fluct, 'h20110': h20110_fluct, 'h11200': h11200_fluct,
                'h20020': h20020_fluct, 'h20200': h20200_fluct, 'h00310': h00310_fluct, 'h00400': h00400_fluct}

    def __getitem__(self, item):
        return self.terms[item]

    def __str__(self):
        text = f'nonlinear terms: {self.n_periods:d} periods\n'
        for i, j in self.terms.items():
            text += f'    {str(i):7}: {abs(j):.2f}\n'
        return text
