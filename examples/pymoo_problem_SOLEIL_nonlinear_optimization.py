"""SOLEIL nonlinear optimization example.

This module defines a multi-objective optimization problem for the SOLEIL
storage ring using pymoo.

Example:

    problem = MyProblem()
    pop_size = 5000
    max_gen = 100
    algorithm = MyAlgorithm(pop_size=pop_size)

    res = minimize(
        problem,
        algorithm,
        ("n_gen", max_gen),
        seed=7,
        callback=MyCallback(),
        verbose=True,
    )
"""

import os

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
import multiprocessing as mp
from pymoo.operators.sampling.rnd import FloatRandomSampling
import numpy as np
import simplestoragering as ssr
from simplestoragering.objectives import quantify_rdt_fluctuation


def compute(v):
    """calculate objectives.
    return [obj0, obj1], [constraints]. constraints should be < 0.
    """
    try:
        ring, k2s, k1s = generate_ring(v)
    except ssr.Unstable:
        return 1e10, 1e10
    rdts = ring.driving_terms(verbose=False)
    RDT_fluct_0 = quantify_rdt_fluctuation(rdts, w=1 / 200)
    dQxx = - np.real(rdts['h2200']) * 4 / np.pi
    dQxy = - np.real(rdts['h1111']) * 2 / np.pi
    dQyy = - np.real(rdts['h0022']) * 4 / np.pi  
    betax = ring.elements[0].betax
    betay = ring.elements[0].betay
    jx = (0.02 ** 2 / betax / 2)
    jy = (0.006 ** 2 / betay / 2)

    tune_1 = (ring.nux - 18 + jx * dQxx, ring.nuy - 10 + jx * dQxy)  # linear tune shifts at x=20 mm
    tune_2 = (ring.nux - 18 + jy * dQxy, ring.nuy - 10 + jy * dQyy)  # linear tune shifts at y=6 mm.

    unstable = -0.1
    RDT_fluct = np.zeros(5)
    nux = np.zeros(5)
    nuy = np.zeros(5)
    for i, dp in enumerate([-0.06, -0.05, -0.04, 0.03, 0.04]):
        try:
            ring.off_momentum_optics(delta=dp)
            betax = ring.elements[0].betax
            rdts = ring.driving_terms(verbose=False)
            RDT_fluct[i] = quantify_rdt_fluctuation(rdts, w=0.015 / betax ** 0.5) ** 2 / betax
            nux[i] = ring.elements[-1].nux - 18
            nuy[i] = ring.elements[-1].nuy - 10
        except ssr.Unstable:
            unstable += 1
    return [RDT_fluct_0, np.average(RDT_fluct)**0.5], [max(np.abs(k2s)) - 73,  # maximum strength
                                                       unstable,  # stable periodic solutions from delta=-6% to 4%
                                                       nux[0] - 2 * nuy[0], # tune shifts with momentum don't cross nux-2nuy
                                                       tune_1[0] - tune_1[1],  # tune shifts with amplitude don't cross nux-nuy
                                                       tune_2[0] - tune_2[1]]


class MyProblem(Problem):
    def __init__(self):
        lb = [6, -35, -30.0, 30.0, -60, 35., -73, 47., 5, -0, 18.1, 10.15]
        ub = [26., -10, -10.0, 56, -45, 50, -60, 62, 20., 0., 18.2, 10.24]
        # lb = [0.0, -73, -73, 0., -73, 0., -73, 0., 0.0, -0, 18.1, 10.1]
        # ub = [73., 0.0, 0.0, 73, 0.0, 73, 0.0, 73, 73., 0., 18.4, 10.4]
        super().__init__(n_var=12, n_obj=2, n_constr=5, xl=np.array(lb), xu=np.array(ub))

    def _evaluate(self, x, out, *args, **kwargs):
        nonlinears = np.zeros((len(x), self.n_obj))
        CV = np.zeros((len(x), self.n_constr))
        pool = mp.Pool(processes=int(4))
        results = pool.map(compute, x)
        pool.close()
        pool.join()
        for i, res in enumerate(results):
            nonlinears[i, :] = res[0]
            CV[i, :] = res[1]
        out['F'] = nonlinears
        out['G'] = CV
        # warnings.warn('Not parallel!!!!')  # debug with the non-parallel version.
        # for i, v in enumerate(x):
        #     res = compute(v)
        #     nonlinears[i, :] = res[0]
        #     CV[i, :] = res[1]
        # out['F'] = nonlinears
        # out['G'] = CV


class MyCallback(Callback):

    def __init__(self):
        super().__init__()

    def notify(self, algorithm):
        # print(time.strftime('%H:%M:%S'), f', {algorithm.n_gen}.\n')
        if (algorithm.n_gen % 10 == 0) or (2 > algorithm.n_gen):
            dir_name = f'soleil_1_3_4/generations/gen{algorithm.n_gen}'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            pop = algorithm.pop
            x = pop.get('X')
            obj = pop.get('F')
            cv = pop.get('G')
            if obj is not None:
                np.savetxt(dir_name + '/Obj.csv', obj, delimiter=',')
            if cv is not None:
                np.savetxt(dir_name + '/Constr.csv', cv, delimiter=',')
            if x is not None:
                np.savetxt(dir_name + '/Var.csv', x, delimiter=',')
        # pass


class MyAlgorithm(NSGA2):

    def __init__(self, pop_size=None, sampling=FloatRandomSampling(), p_mutation=0.2, p_crossover=0.8):
        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         crossover=SBX(eta=15, prob=p_crossover),
                         mutation=PM(eta=20, prob=p_mutation))


def generate_ring(v) -> [ssr.CSLattice, np.ndarray, np.ndarray]:
    """generate SOLEIL nonlinear solution"""
    # GLOBVAL.E0 = 2.7391e9  # %Ring energy
    # GLOBVAL.LatticeFile = mfilename
    # FAMLIST = cell(0)
    ssr.set_ref_energy(2739.1)
    # disp(['** Loading SOLEIL magnet lattice ', mfilename])
    #
    # % L0 = 3.5409702042030095e+02% design length [m]
    # % C0 = 2.99792458e8          % speed of light [m/s]
    # L0 = 3.540970204203006e+02  # % correction to get a true zero H-closed orbit
    # C0 =  PhysConstant.speed_of_light_in_vacuum.value          % speed of light [m/s]
    # HarmNumber = 416
    #
    # %% RF Cavity
    # %              NAME   L     U[V]       f[Hz]          h        method
    # CAV = rfcavity('RF' , 0.0 , 2.6e+6 , HarmNumber*C0/L0, HarmNumber ,'RFCavityPass')
    CAV = ssr.Mark('RF')  # add a mark to make sure the magnet layout is correct.

    # %% Marker and apertures
    SECT1 = ssr.Mark('SECT1')
    SECT2 = ssr.Mark('SECT2')
    SECT3 = ssr.Mark('SECT3')
    SECT4 = ssr.Mark('SECT4')
    DEBUT = ssr.Mark('DEBUT')
    FIN   = ssr.Mark('FIN')
    #
    # %% FBT Elements
    STRIPLINE_H = ssr.Mark('STRIPLINE_H')
    STRIPLINE_V = ssr.Mark('STRIPLINE_V')
    STRIPLINE_CC = ssr.Mark('STRIPLINE_CC')
    #
    # %% MIK ssr.Marks
    mSDC   = ssr.Mark('mSDC')  # % beginning/end SDC
    mMIKTAP_seg3 = ssr.Mark('mMIKTAP_seg3')
    mMIKTAP_seg2 = ssr.Mark('mMIKTAP_seg2')
    mMIKTAP_seg1 = ssr.Mark('mMIKTAP_seg1')
    mMIKABS = ssr.Mark('mMIKABS')
    mMIKTAP = ssr.Mark('mMIKTAP')
    mSDM   = ssr.Mark('mSDM')  # % beginning/end SDM
    mcSDM  = ssr.Mark('mcSDM')  #%center SDM
    mFFWRCOR_special = ssr.Mark('mFFWRCOR_special')
    mSDNANO3_special = ssr.Mark('SDNANO3_special')

    mROCK = ssr.Mark('mROCK')

    mHU36 = ssr.Mark('mHU36')

    # %% SCRAPER
    HSCRAP = ssr.Mark('HSCRAP')
    VSCRAP = ssr.Mark('VSCRAP')
    #
    # %INJ = aperture('INJ',[-0.035 0.035 -0.0125 0.0125]*100,'AperturePass')
    #
    # %% Elements in Injection section
    PtINJ = ssr.Mark('PtINJ')
    K1 = ssr.Mark('K1')
    K2 = ssr.Mark('K2')
    K3 = ssr.Mark('K3')
    K4 = ssr.Mark('K4')

    # %% BPM
    BPM =  ssr.Mark('BPM')
    BPM_DOUBLE = ssr.Mark('BPM_DOUBLE')

    # %% XBPM                                                                          <                                                                                    -
    PXBPM = ssr.Mark('PXBPM')

    # %% QUADRUPOLES (compensation de l'effet des defauts de focalisation des
    LQC = 0.180100E+00*2
    LQL = 0.248100E+00*2

    Q1 = ssr.Quadrupole('Q1', LQC, -1.210124e+00, n_slices=20)
    Q2 = ssr.Quadrupole('Q2', LQL,  1.687181e+00, n_slices=20)
    Q3 = ssr.Quadrupole('Q3', LQC, -6.375736e-01, n_slices=20)
    Q4 = ssr.Quadrupole('Q4', LQC, -1.229389e+00, n_slices=20)
    Q5 = ssr.Quadrupole('Q5', LQC,  1.713696e+00, n_slices=20)
    Q6 = ssr.Quadrupole('Q6', LQC, -1.163583e+00, n_slices=20)
    Q7 = ssr.Quadrupole('Q7', LQL,  2.035259e+00, n_slices=20)
    Q8 = ssr.Quadrupole('Q8', LQC, -1.357527e+00, n_slices=20)
    Q9 = ssr.Quadrupole('Q9', LQC, -1.338309e+00, n_slices=20)
    Q10 = ssr.Quadrupole('Q10', LQC,  1.736498e+00, n_slices=20)
    Q11 = ssr.Quadrupole('Q11', LQC, -1.671113e+00, n_slices=20)
    Q12 = ssr.Quadrupole('Q12', LQL,  1.667791e+00, n_slices=20)

    # %% SEXTUPOLES CHROMATICITES NULLES dans TracyII
    # %avec defauts de focalisation des dipoles
    # %P. Brunelle 02/05/06
    LSext = 0.16
    n_slices = 8
    # S1 = ssr.Sextupole('S1', LSext, 7.63643672 * 2, n_slices=n_slices)
    # S2 = ssr.Sextupole('S2', LSext, -13.086174 * 2, n_slices=n_slices)
    # S3 = ssr.Sextupole('S3', LSext, -11.85318 * 2, n_slices=n_slices)
    # S4 = ssr.Sextupole('S4', LSext, 22.3432283 * 2, n_slices=n_slices)
    # S5 = ssr.Sextupole('S5', LSext, -24.812971 * 2, n_slices=n_slices)
    # S6 = ssr.Sextupole('S6', LSext, 23.4878817 * 2, n_slices=n_slices)
    # S7 = ssr.Sextupole('S7', LSext, -32.61828 * 2, n_slices=n_slices)
    # S8 = ssr.Sextupole('S8', LSext, 25.7909823 * 2, n_slices=n_slices)
    # S9 = ssr.Sextupole('S9', LSext, -23.508458 * 2, n_slices=n_slices)
    # S10 = ssr.Sextupole('S10', LSext, 13.4674029 * 2, n_slices=n_slices)
    # S11 = ssr.Sextupole('S11', LSext, 8.48081452 * 2, n_slices=n_slices)
    # S12 = ssr.Sextupole('S12', LSext, 1.000E-10 * 2, n_slices=n_slices)
    S1 = ssr.Sextupole('S1', LSext, v[0], n_slices=n_slices)
    S2 = ssr.Sextupole('S2', LSext, v[1], n_slices=n_slices)
    S3 = ssr.Sextupole('S3', LSext, v[2], n_slices=n_slices)
    S4 = ssr.Sextupole('S4', LSext, v[3], n_slices=n_slices)
    S5 = ssr.Sextupole('S5', LSext, v[4], n_slices=n_slices)
    S6 = ssr.Sextupole('S6', LSext, v[5], n_slices=n_slices)
    S7 = ssr.Sextupole('S7', LSext, v[6], n_slices=n_slices)
    S8 = ssr.Sextupole('S8', LSext, v[7], n_slices=n_slices)
    S9 = ssr.Sextupole('S9', LSext, 0.0, n_slices=n_slices)
    S10 = ssr.Sextupole('S10', LSext, 0.0, n_slices=n_slices)
    S11 = ssr.Sextupole('S11', LSext, v[8], n_slices=n_slices)
    S12 = ssr.Sextupole('S12', LSext, v[9], n_slices=n_slices)

    # % Compensation chromaticit? Superbend ROCK
    # %S9  =  sextupole('S9' , LSext, -370019115*1e-8/LSext)
    # %S10 =  sextupole('S10', LSext,  231833387*1e-8/LSext)

    # %% Skew quadrupoles
    # SQPassMethod = SPassMethod
    LQT = 1e-8
    # QT  =  skewquad('SkewQuad', LQT, 0.0, SQPassMethod)
    # QTPX2    =  skewquad('QTPX2', 1e-10, 0.0, SPassMethod)  # % PX2
    QT = ssr.Drift('QT', LQT)
    QTPX2 = ssr.Drift('QTPX2', 1e-10)
    #
    # %% Machine study kickers
    KEMH = ssr.Mark('KEMH')
    KEMV = ssr.Mark('KEMV')
    KEMV2 = ssr.Mark('KEMV2')
    #
    # %% Multipole Injection Kickers as a thin element
    MIK = ssr.Mark('MIK')
    mMIK = ssr.Mark('mMIK')
    #
    # %% PX2C H-correctors
    # % Tuners
    PX2 = ssr.Mark('PX2')
    # % Main magnets
    CHIPX2D1 = ssr.HBend('CHIPX2D1', 0.026, -2.25e-3,  0.00e-3, -2.25e-3, n_slices=1)
    CHIPX2D2 = ssr.HBend('CHIPX2D2', 0.052,  4.50e-3, -2.25e-3,  2.25e-3, n_slices=1)
    CHIPX2D3 = ssr.HBend('CHIPX2D3', 0.026, -2.25e-3,  2.25e-3,  0.00e-3, n_slices=1)
    PX2C= [QTPX2, PX2]

    # %% NANOC magnets for nanoscopium
    # % Tuners
    CHINANO = ssr.Mark('CHINANO')
    # % Main magnets
    CHINANOD1 = ssr.HBend('CHINANOD1', 0.069, -0.50e-3,  0.00e-3, -0.50e-3, n_slices=1)
    CHINANOD2 = ssr.HBend('CHINANOD2', 0.069, -5.38e-3, -0.50e-3, -5.88e-3, n_slices=1)
    CHINANOD3 = ssr.HBend('CHINANOD3', 0.138, 11.88e-3, -5.88e-3, +6.00e-3, n_slices=1)
    CHINANOD4 = ssr.HBend('CHINANOD4', 0.069, -6.00e-3,  6.00e-3,  0.00e-3, n_slices=1)
    #
    # %% HU640
    HCMHU640 =  ssr.Mark('HCMHU640')
    VCMHU640 =  ssr.Mark('VCMHU640')

    # %% Fast feedback correctors
    FCOR =  ssr.Mark('FCOR')
    #
    # %% Feedforward correctors
    FFWDCOR = ssr.Mark('FFWDCOR')
    FFWDCOR_special = ssr.Drift('FFWDCOR_special',10e-2)
    #
    # %% Slow correctors in sextupole magnets
    # % With thin sextupoles : SXi = [Si COR QT] -> length of SXi in this case is
    # % L_thinSXi = 1e-8 + 0 +1e-8, with i = 1,...,12
    #
    L_thinSXi = 2e-8
    SX1  = [S1,  QT]
    SX2  = [S2,  QT]
    SX3  = [S3,  QT]
    SX4  = [S4,  QT]
    SX5  = [S5,  QT]
    SX6  = [S6,  QT]
    SX7  = [S7,  QT]
    SX8  = [S8,  QT]
    SX9  = [S9,  QT]
    SX10 = [S10, QT]
    SX11 = [S11, QT]
    SX12 = [S12, QT]

    # %% DIPOLES
    # % {** 1.3815 factor to fit with BETA ??? strange **}
    # %theta = 2*pi/32
    # %fullgap = 0.105*0.724*2/6*1.3815*0.
    # % BEND  =  rbend2('BEND', L, theta, theta/2, theta/2, 0.0, ...
    # %                 fullgap,'BendLinearFringeTiltPass')
    theta = 2 * np.pi/32
    # %theta2 = theta/2
    thetae = theta/2 - 0.6e-3
    thetas = theta/2 + 0.9e-3
    gap = 0.037 * 0.724 * 2
    # gap = 0.0
    K = 0.00204
    # fullgap = 0.037*0.724*2
    # nn=10
    #
    # BEND  =  rbend3('BEND', 1.05243, theta, thetae, thetas, K,fullgap,1,1,'BndMPoleSymplectic4Pass')
    BEND  = ssr.HBend('BEND', 1.05243, theta, thetae, thetas, K, gap, n_slices=4)
    # % BEND  =  rbend2('BEND', 1.05243, theta, thetae, thetas, K,fullgap,'BndMPoleSymplectic4Pass')

    # % SUPERBEND liced in 7 parts
    theta1 = 0.486019 * np.pi / 180
    theta2 = 0.478493 * np.pi / 180
    theta3 = 0.354950 * np.pi / 180
    theta4 = 0.415154 * np.pi / 180
    theta5 = 0.689833 * np.pi / 180
    theta6 = 8.317505 * np.pi / 180
    theta7 = 0.508045 * np.pi / 180

    long1 = 0.11350
    long2 = 0.02790
    long3 = 0.02090
    long4 = 0.06090
    long5 = 0.09180
    long6 = 0.64563
    long7 = 0.09180
    #
    #  % fringe field and edge focusing set to zero.
    #  % Only a normal gradient component
    quad_on = 1
    k1 = -0.012838866*quad_on/long1
    k2 =  0.012039858*quad_on/long2
    k3 = -0.005363209*quad_on/long3
    k4 =  0.006238835*quad_on/long4
    k5 = -0.011328412*quad_on/long5
    k6 = -0.03331757*quad_on/long6
    k7 = -0.0146339*quad_on/long7
    #
    # % BEND only on first slice for MML compatibility
    # ROCK1 = ssr.HBend('ROCK1', long1, theta1, 0, 0, k1)
    # ROCK2 = ssr.HBend('ROCK2', long2, theta2, 0, 0, k2)
    # ROCK3 = ssr.HBend('ROCK3', long3, theta3, 0, 0, k3)
    # ROCK4 = ssr.HBend('ROCK4', long4, theta4, 0, 0, k4)
    # ROCK5 = ssr.HBend('ROCK5', long5, theta5, 0, 0, k5)
    # ROCK6 = ssr.HBend('ROCK6', long6, theta6, 0, 0, k6)
    # ROCK7 = ssr.HBend('ROCK7', long7, theta7, 0, 0, k7)
    #
    # % set SUPERBEND composantes sextupolaires
    # # kick-drift-kick
    # sextu_on = 1
    # ksext_l = 1e-8
    # ksext1 = -1.629113152 * sextu_on / ksext_l  # * 2 / 2
    # ksext2 = -7.349588875 * sextu_on / ksext_l  # * 2 / 2
    # ksext3 = -5.380262179 * sextu_on / ksext_l  # * 2 / 2
    # ksext4 = -1.552736673 * sextu_on / ksext_l  # * 2 / 2
    # ksext5 = -0.211857715 * sextu_on / ksext_l  # * 2 / 2
    # ksext6 = -1.397225492 * sextu_on / ksext_l  # * 2 / 2
    # ksext7 = -0.22212443 * sextu_on / ksext_l  # * 2 / 2
    # ROCK1_sext = ssr.Sextupole('ROCK1_sext', ksext_l, ksext1, n_slices=1)
    # ROCK12_sext = ssr.Sextupole('ROCK2_sext', ksext_l, ksext1 + ksext2, n_slices=1)
    # ROCK23_sext = ssr.Sextupole('ROCK2_sext', ksext_l, ksext2 + ksext3, n_slices=1)
    # ROCK34_sext = ssr.Sextupole('ROCK3_sext', ksext_l, ksext3 + ksext4, n_slices=1)
    # ROCK45_sext = ssr.Sextupole('ROCK4_sext', ksext_l, ksext4 + ksext5, n_slices=1)
    # ROCK56_sext = ssr.Sextupole('ROCK5_sext', ksext_l, ksext5 + ksext6, n_slices=1)
    # ROCK67_sext = ssr.Sextupole('ROCK6_sext', ksext_l, ksext6 + ksext7, n_slices=1)
    # ROCK7_sext = ssr.Sextupole('ROCK7_sext', ksext_l, ksext7, n_slices=1)
    # ROCK = [ROCK1_sext, ROCK1, ROCK12_sext, ROCK2, ROCK23_sext, ROCK3, ROCK34_sext,
    #                     ROCK4, ROCK45_sext, ROCK5, ROCK56_sext, ROCK6, ROCK67_sext, ROCK7, ROCK7_sext]

    # % ROCK = BEND
    #
    # % set SUPERBEND composantes sextupolaires
    sextu_on = 1;
    ksext1 = -1.629113152 / long1 * sextu_on;
    ksext2 = -7.349588875 / long2 * sextu_on;
    ksext3 = -5.380262179 / long3 * sextu_on;
    ksext4 = -1.552736673 / long4 * sextu_on;
    ksext5 = -0.211857715 / long5 * sextu_on;
    ksext6 = -1.397225492 / long6 * sextu_on;
    ksext7 = -0.22212443 / long7 * sextu_on;
    ROCK1 = ssr.HBend('ROCK1', long1, theta1, 0, 0, k1, k2=ksext1 * 2, n_slices=1)
    ROCK2 = ssr.HBend('ROCK2', long2, theta2, 0, 0, k2, k2=ksext2 * 2, n_slices=1)
    ROCK3 = ssr.HBend('ROCK3', long3, theta3, 0, 0, k3, k2=ksext3 * 2, n_slices=1)
    ROCK4 = ssr.HBend('ROCK4', long4, theta4, 0, 0, k4, k2=ksext4 * 2, n_slices=1)
    ROCK5 = ssr.HBend('ROCK5', long5, theta5, 0, 0, k5, k2=ksext5 * 2, n_slices=1)
    ROCK6 = ssr.HBend('ROCK6', long6, theta6, 0, 0, k6, k2=ksext6 * 2, n_slices=3)
    ROCK7 = ssr.HBend('ROCK7', long7, theta7, 0, 0, k7, k2=ksext7 * 2, n_slices=1)
    ROCK = [ROCK1, ROCK2, ROCK3, ROCK4, ROCK5, ROCK6, ROCK7]
    #
    # FAMLIST{ROCK1}.ElemData.PolynomB(3) = ksext1;
    # FAMLIST{ROCK1}.ElemData.PolynomA(3) = 0.0;
    # FAMLIST{ROCK1}.ElemData.MaxOrder = 2;
    # FAMLIST{ROCK2}.ElemData.PolynomB(3) = ksext2;
    # FAMLIST{ROCK2}.ElemData.PolynomA(3) = 0.0;
    # FAMLIST{ROCK2}.ElemData.MaxOrder = 2;
    # FAMLIST{ROCK3}.ElemData.PolynomB(3) = ksext3;
    # FAMLIST{ROCK3}.ElemData.PolynomA(3) = 0.0;
    # FAMLIST{ROCK3}.ElemData.MaxOrder = 2;
    # FAMLIST{ROCK4}.ElemData.PolynomB(3) = ksext4;
    # FAMLIST{ROCK4}.ElemData.PolynomA(3) = 0.0;
    # FAMLIST{ROCK4}.ElemData.MaxOrder = 2;
    # FAMLIST{ROCK5}.ElemData.PolynomB(3) = ksext5;
    # FAMLIST{ROCK5}.ElemData.PolynomA(3) = 0.0;
    # FAMLIST{ROCK5}.ElemData.MaxOrder = 2;
    # FAMLIST{ROCK6}.ElemData.PolynomB(3) = ksext6;
    # FAMLIST{ROCK6}.ElemData.PolynomA(3) = 0.0;
    # FAMLIST{ROCK6}.ElemData.MaxOrder = 2;
    # FAMLIST{ROCK7}.ElemData.PolynomB(3) = ksext7;
    # FAMLIST{ROCK7}.ElemData.PolynomA(3) = 0.0;
    # FAMLIST{ROCK7}.ElemData.MaxOrder = 2;

    # %% DRIFT SPACES
    #
    SD1a = ssr.Drift('SD1a',  1.4125)
    SD1b = ssr.Drift('SD1b',  0.7575)
    SD2 = ssr.Drift('SD2',  0.369900 -(LSext + LQT - L_thinSXi)/2)
    SD3 = ssr.Drift('SD3',   0.181900 -(LSext + LQT - L_thinSXi)/2)
    SD5 = ssr.Drift('SD5',  0.179900 -(LSext + LQT - L_thinSXi)/2)
    SD6 = ssr.Drift('SD6',  0.79000 -(LSext + LQT - L_thinSXi)/2)
    SD7 = ssr.Drift('SD7',  0.419900)
    SD8 = ssr.Drift('SD8',  0.1799000 -(LSext + LQT - L_thinSXi)/2)
    SD12= ssr.Drift('SD12', 0.44990 -(LSext + LQT - L_thinSXi)/2)
    SD12u= ssr.Drift('SD12u', (0.36565-LQC/2-0.015))  # % upstream H-scraper C01
    SDHSCRAP = ssr.Drift('SDHSCRAP', 0.03)  # %  SCRAPER LENGTH
    SD12d= ssr.Drift('SD12d', 0.44990-(0.36565-LQC/2) -(LSext + LQT - L_thinSXi)/2-0.015)  # % downstream H-scraper
    SD12u2= ssr.Drift('SD12u2', 0.36565-LQC/2-0.015)  # % upstream H-scraper exterior Q5.1/S4 C16
    SDHSCRAP2 = ssr.Drift('SDHSCRAP2', 0.03)
    SD12d2= ssr.Drift('SD12d2', 0.44990-(0.36565-LQC/2) -(LSext + LQT - L_thinSXi)/2-0.015)  # % H-scraper exterior
    SD1d = ssr.Drift('SD1d',  0.5170)
    SD14a = ssr.Drift('SD14a', 0.38500000)
    SD9a = ssr.Drift('SD9a',  0.204200)
    SD10a = ssr.Drift('SD10a', 0.172300)
    SDAC1 = ssr.Drift('SDAC1', 1.48428)
    #
    # %BEGIN MIK
    SMIKu = ssr.Drift('SMIKu', 1658.9e-3)  # % Normal chamber upstream MIK : previously 1.5159+180e-3 (modified 18.01.21)
    # %SDMIKABS = ssr.Drift('SDMIKABS', 0.04)  # % upstream MIK absorber
    SMIKhalf = ssr.Drift('SMIKhalf', 0.4/2)  # % MIK chamber
    SMIKd = ssr.Drift('SMIKd', 179.46e-3)  # % Normal chamber downstream MIK 370.9e-3  previously 190.9e-3+0.56e-3 (modified 18.01.21)
    # %END MIK
    #
    # % BEGIN around MIK
    MIKTAP_seg3 = ssr.Drift('MIKTAP_seg3',63e-3)
    MIKTAP_seg2 = ssr.Drift('MIKTAP_seg2',60e-3)
    MIKTAP_seg1 = ssr.Drift('MIKTAP_seg1',44e-3)
    MIKABS = ssr.Drift('MIKABS',198.1e-3)
    MIKTAP = ssr.Drift('MIKTAP',167e-3)
    # % END around MIK
    #
    SD13a= ssr.Drift('SD13a', 3.141452)
    SD1e = ssr.Drift('SD1e',  5.6589)
    SD1c1 = ssr.Drift('SD1c1',  0.8410)  # % K3 - FCOR
    SD1c2 = ssr.Drift('SD1c2',  0.601)  # % FCOR KEMH
    SD1c3u= ssr.Drift('SD1c3u', 0.683)  # % KEMH - VSCRAPER
    SD1c3d= ssr.Drift('SD1c3d', 1.560-0.683)  # % VSCRAPER - K4
    SD91 = ssr.Drift('SD91',  0.251240)
    SD41 = ssr.Drift('SD41',  0.2521 -(LSext + LQT - L_thinSXi)/2)
    SD42 = ssr.Drift('SD42',  0.205 -(LSext + LQT - L_thinSXi)/2)
    SD92 = ssr.Drift('SD92',  0.204300)
    SD93 = ssr.Drift('SD93',  0.251300)
    SD43 = ssr.Drift('SD43', 0.2051 -(LSext + LQT - L_thinSXi)/2)
    SD141 = ssr.Drift('SD141', 0.431900)
    SDB1 = ssr.Drift('SDB1', 0.29100)
    SDB2 = ssr.Drift('SDB2', 0.16680000)
    SDB3 = ssr.Drift('SDB3', 0.252 -(LSext + LQT - L_thinSXi)/2)
    SDB4 = ssr.Drift('SDB4', 0.2776 -(LSext + LQT - L_thinSXi)/2)
    SDB5 = ssr.Drift('SDB5', 0.205 -(LSext + LQT - L_thinSXi)/2)
    SDB6 = ssr.Drift('SDB6', 0.119800)
    SDB7 = ssr.Drift('SDB7', 0.166900)
    SDB8 = ssr.Drift('SDB8', 0.252 -(LSext + LQT - L_thinSXi)/2)
    SDB9 = ssr.Drift('SDB9', 0.119800)
    SDB10= ssr.Drift('SDB10',0.166900)
    SDB11= ssr.Drift('SDB11',0.2519 -(LSext + LQT - L_thinSXi)/2)
    SDB12= ssr.Drift('SDB12',0.2049 -(LSext + LQT - L_thinSXi)/2)
    SDB13= ssr.Drift('SDB13',0.119800)
    SDB14= ssr.Drift('SDB14',0.1668000)
    SDB15= ssr.Drift('SDB15',0.252 -(LSext + LQT - L_thinSXi)/2)
    SDB17= ssr.Drift('SDB17',0.205 -(LSext + LQT - L_thinSXi)/2)
    SDB18= ssr.Drift('SDB18',0.1199000)
    SDC1 = ssr.Drift('SDC1' , 0.241900)
    SDC2 = ssr.Drift('SDC2' , 0.079)
    SDC3 = ssr.Drift('SDC3' , 0.07845)
    SDC4 = ssr.Drift('SDC4' , 0.3358 -(LSext + LQT - L_thinSXi)/2)
    SDC5 = ssr.Drift('SDC5' , 0.0846)
    SDC6 = ssr.Drift('SDC6' , 0.079)
    SDC7 = ssr.Drift('SDC7' , 0.342 -(LSext + LQT - L_thinSXi)/2 )
    SDC8 = ssr.Drift('SDC8' , 0.241900 )
    SDC9 = ssr.Drift('SDC9' , 0.079  )
    DRFT10= ssr.Drift('DRFT10',0.07845)
    DRFT11= ssr.Drift('DRFT11',0.2419000  )
    DRFT12= ssr.Drift('DRFT12',0.3358 -(LSext + LQT - L_thinSXi)/2 )
    DRFT13= ssr.Drift('DRFT13',0.0846 )
    DRFT14= ssr.Drift('DRFT14',0.0788 )
    DRFT15= ssr.Drift('DRFT15',0.3422 -(LSext + LQT - L_thinSXi)/2 )
    DRFT16= ssr.Drift('DRFT16',0.241900  )
    DRFT17= ssr.Drift('DRFT17',0.079  )
    DRFT18= ssr.Drift('DRFT18',0.07845)
    DRFT19= ssr.Drift('DRFT19',0.24190  )
    SDC20= ssr.Drift('SDC20',0.241900  )
    SDC21= ssr.Drift('SDC21',0.079  )
    SDC22= ssr.Drift('SDC22',0.29090)
    SDC24= ssr.Drift('SDC24',1.379)
    # %SDC23a= ssr.Drift('SDC23a',0.632  )  # % BPM - K1
    # %SDC23b= ssr.Drift('SDC23b',1.983  )  # % K1 - KEMV
    SDC23bu= ssr.Drift('SDC23bu',1.983/2-0.084/2)  # % for SCRAPERV
    SDSCRAPV= ssr.Drift('SDSCRAPV', 0.084)  # % SCRAPERV
    SDC23c= ssr.Drift('SDC23c',1.019  )  # % KEMV - K2
    SDC23d= ssr.Drift('SDC23d',0.676  )  # % K2 - FCOR
    SDC23e= ssr.Drift('SDC23e',0.147  )  # % BPM - FCOR [1 1]
    SDC23f= ssr.Drift('SDC23f',0.485  )  # % FCOR [1 1] next SD
    #
    # % HU640 straight section
    SDHU640a = ssr.Drift('SDHU640a',  1.7394)
    SDHU640b = ssr.Drift('SDHU640b',  0.6400)
    SDHU640c = ssr.Drift('SDHU640c',  3.2795)
    SDHU640d = ssr.Drift('SDHU640d',  3.1195)
    SDHU640e = ssr.Drift('SDHU640e',  0.6400)
    SDHU640f = ssr.Drift('SDHU640f',  1.8994)
    #
    # % PX2 straights
    CHIPX2D1_length = 0.026
    CHIPX2D2_length = 0.052
    SDPX2a= ssr.Drift('SDPX2a', 0.363902-CHIPX2D1_length/2)  # % BPM - CHI.1
    SDPX2b= ssr.Drift('SDPX2b', 2.857550-CHIPX2D1_length/2-CHIPX2D2_length/2)  # % CHI.1 - CHI.2
    SDPX2c= ssr.Drift('SDPX2c', 0.203902-CHIPX2D1_length/2)  # % CHI.3 - BPM
    #
    # % Nanoscopium straigths (upstream)
    CHINANOD1_length = 0.069
    SDNANO1 = ssr.Drift('SDNANO1',  0.4501-CHINANOD1_length/2)  # % BPM - CHI.1
    SDNANO2 = ssr.Drift('SDNANO2',  0.5529-CHINANOD1_length/2)  # % CHI.1 - FFWDCOR
    SDNANO3 = ssr.Drift('SDNANO3',  2.5630)  # % FFWDCOR - FFWDCOR
    SDNANO3_special = ssr.Drift('SDNANO3_special',  2.5630-10e-2)  # % FFWDCOR - FFWDCOR
    SDNANO4 = ssr.Drift('SDNANO4',  0.4330-CHINANOD2.length/2)  # % FFWDCOR - CHI.2
    SDNANO5 = ssr.Drift('SDNANO5',  0.2683-CHINANOD2.length/2)  # % CHI.2 - BPM
    SDNANO6 = ssr.Drift('SDNANO6',  0.0780)  # % BPM - FCOR
    SDNANO6a= ssr.Drift('SDNANO6a', 0.5017-0.0780-Q11.length/2)  # % BPM - Q11.1
    SDNANO7 = ssr.Drift('SDNANO7',  0.4100-Q11.length/2-S12.length/2)  # % Q11.1 - S12
    SDNANO8 = ssr.Drift('SDNANO8',  0.4800-Q12.length/2-S12.length/2)  # % S12 - Q11
    # % Nanoscopium straigths (downstream)
    SDNANO9 = ssr.Drift('SDNANO9',  0.4628-0.0780-Q11.length/2)  # % Q11.2 - FOFB
    SDNANO10= ssr.Drift('SDNANO10', 0.3072-CHINANOD3.length/2)  # % BPM - CHI.3
    SDNANO11= ssr.Drift('SDNANO11', 0.4330-CHINANOD3.length/2)  # % FFWDCOR - FFWDCOR
    #
    # % For FBT - SDM07
    to_BPM_double_H = ssr.Drift('to_BPM_double_H', 3.141452*2 - 0.142 - 0.4610 - 5.47920)
    to_BPM_double_H2 = ssr.Drift('to_BPM_double_H2', 0.142)
    to_stripH = ssr.Drift('to_stripH', 0.4610 )
    SDM07_reste = ssr.Drift('SDM07_reste', 5.47920 )
    #
    # % For FBT - SDL09
    to_BPM_double_CC = ssr.Drift('to_BPM_double_CC', 5.6589 - 1.3191)
    to_BPM_double_CC2 = ssr.Drift('to_BPM_double_CC2', 0.030 )
    to_stripCC = ssr.Drift('to_stripCC', 0.2510 )
    to_stripV = ssr.Drift('to_stripV', 0.5450 )
    SD1e_reste = ssr.Drift('SD1e_reste', 0.4931)
    #
    # % For FBT - SD6
    to_BPM_double_V = ssr.Drift('to_BPM_double_V', 0.248 + L_thinSXi/4-LSext/2)
    to_BPM_double_V2 = ssr.Drift('to_BPM_double_V2', 0.142 )
    SD6_reste = ssr.Drift('SD6_reste',  0.79000 - 0.39)
    SD6_BPM_double = [to_BPM_double_V, BPM_DOUBLE, to_BPM_double_V2, BPM_DOUBLE, SD6_reste]

    # %% STRAIGHT SECTIONS (between BPMs)
    # % 4 long straight sections (12 m, available part 10.50 m)
    #
    # %SDL01 (injection) is split in upstream and downstream parts
    SDL01d = [SD1a, PtINJ,   SD1b,   K3,  SD1c1, FCOR,  SD1c2,  KEMH, SD1c3u, KEMV2, SD1c3d, K4, SD1d]
    SDL01u = [SDC23e,  FCOR, SDC23f, K1, SDC23bu, VSCRAP, SDSCRAPV, VSCRAP, SDC23bu, KEMV, SDC23c, K2, SDC23d, SDC24]
    # % SDL05 HU640 straight section
    HU640upstream   = [SDHU640a, VCMHU640, SDHU640b, HCMHU640, SDHU640c]
    HU640downstream = [SDHU640d, HCMHU640, SDHU640e, VCMHU640, SDHU640f]
    SDL05  = HU640upstream + HU640downstream  # % DESIRS HU640
    # % LONG STRAIGHT SECTION WITH FBT system
    # %SDL09  = [SD1e SD1e];
    SDL09  = [SD1e, to_BPM_double_CC, BPM_DOUBLE, to_BPM_double_CC2, BPM_DOUBLE, to_stripCC, STRIPLINE_CC, to_stripV, STRIPLINE_V, SD1e_reste]
    #
    # % TOMOGRAPHY U18 CRYO + NANOSCOPIUM U20
    SDL13u = [SDNANO1, CHINANO, CHINANOD1, SDNANO2, FFWDCOR, SDNANO3, FFWDCOR, SDNANO4, CHINANOD2, CHINANO, SDNANO5, BPM, SDNANO6, FCOR, SDNANO6a, Q11, SDNANO7, SX12, SDNANO8]
    SDL13d = [SDNANO8, SX12, SDNANO7, Q11, SDNANO9, FCOR, SDNANO6, BPM, SDNANO10, CHINANO, CHINANOD3, SDNANO11, mFFWRCOR_special, FFWDCOR_special, mSDNANO3_special, SDNANO3_special, FFWDCOR, SDNANO2, CHINANOD4, CHINANO, SDNANO1]
    SDL13 = SDL13u + [Q12] + SDL13d
    # %SDL13  = [SD1e SD1e]; % NANOSCOPIUM U20 + TOMOGRAPHY U18 CRYO
    #
    # % 12 medium straigt sections (7 m, available part for IDs 5.46 m)
    SDM02 = [mSDM, SD13a, mcSDM, CAV, SD13a, mSDM]  # CRYOMODULE #2
    # %SDM02 = [mSDM SD13ad SD13ad SD13ad SD13ad SD13ad SD13ad SD13ad SD13ad SD13ad SD13ad CAV SD13a mSDM];  # CRYOMODULE #2
    SDM03 = [mSDM, SD13a, mcSDM, SD13a, mSDM]  # CRYOMUDULE #1 not put in the model for simplicity
    SDM04 = [mSDM, SD13a, SD13a, mSDM]  # PLEIADES HU256 + HU80
    SDM06 = [mSDM, SD13a, SD13a, mSDM]  # PUMA future Wiggler
    # %SDM06 = [SDWSV50 SWSV50 SDWSV50];  # PUMA future Wiggler
    # %SDM07 = [mSDM SD13a SD13a mSDM];  # DEIMOS HU52+EMPHU65
    SDM07 = [mSDM, to_BPM_double_H, BPM_DOUBLE, to_BPM_double_H2, BPM_DOUBLE, to_stripH, STRIPLINE_H, SDM07_reste, mSDM]  # DEIMOS HU52+EMPHU65 + FBT
    SDM08 = [mSDM, SD13a, SD13a, mSDM]  # TEMPO HU80+HU44
    SDM10 = [mSDM, SD13a, SD13a, mSDM]  # HERMES HU64+HU42
    SDM11 = [mSDM, SDPX2a, PX2C, CHIPX2D1, SDPX2b, PX2C, CHIPX2D2, SDPX2b, CHIPX2D3, PX2C, SDPX2c, mSDM]  # PX2 U24
    SDM12 = [mSDM, SD13a, SD13a, mSDM]  # ANTARES HU256 + HU60
    SDM14 = [mSDM, SD13a, SD13a, mSDM]  # SEXTANTS (ex microFocus) HU44 + HU80
    SDM15 = [mSDM, SD13a, SD13a, mSDM]  # CASSIOPEE HU256 + HU80
    SDM16 = [mSDM, SD13a, SD13a, mSDM]  # LUCIA HU52
    #
    # % 8 short straight sections (3.6 m, available part for IDs 2.8 m)
    #
    # %SDC02 = [mSDC SDAC1 SMIKu SDMIKABS mMIK SMIKhalf MIK SMIKhalf mMIK SMIKd mSDC];
    #
    SDC02 = [mSDC, SMIKu, mMIKTAP_seg3, MIKTAP_seg3, mMIKTAP_seg2, MIKTAP_seg2, mMIKTAP_seg1, MIKTAP_seg1, mMIKABS, MIKABS, mMIK, SMIKhalf, MIK, SMIKhalf, mMIK, MIKABS, mMIKTAP, MIKTAP, mSDC, SMIKd]

    SDC03 = [mSDC, SDAC1, SDAC1, mSDC]  # % PSICHE WSV50
    SDC06 = [mSDC, SDAC1, SDAC1, mSDC]  # % CRISTAL U20
    SDC07 = [mSDC, SDAC1, SDAC1, mSDC]  # % GALAXIES U20
    SDC10 = [mSDC, SDAC1, SDAC1, mSDC]  # % PX1 U20
    SDC11 = [mSDC, SDAC1, SDAC1, mSDC]  # % SWING U20
    SDC14 = [mSDC, SDAC1, SDAC1, mSDC]  # % SIXS U20
    SDC15 = [mHU36, SDAC1, SDAC1, mHU36]  # % SIRIUS HU36

    # %%Lattice
    # %Superperiods
    #
    # %SUPERPERIOD  # 1
    SUP1 = [BPM, SDB1, Q1, SD2, SX1, SD3, Q2, SDB2, BPM, SD14a, Q3, SD5, SX2, SD6, BEND, SD7, Q4, SD8, SX3, SDB3, BPM, SD9a, Q5, SD12u, HSCRAP, SDHSCRAP, HSCRAP, SD12d, SX4, SDB4, BPM, SD10a, Q5, SD91, BPM, SDB5, SX3, SD8, Q4, SD7, PXBPM, BEND, SD7, Q6, SD5, SX5, SD41, BPM, SDB6, Q7, SD3, SX6, SD2, Q8, SDC1, FCOR, SDC2, BPM, SDM02, BPM, SDC3, FCOR, SDC1, Q8, SD2, SX8, SD3, Q7, SDB7, BPM, SD42, SX7, SD5, Q6, SD7, BEND, SD7, Q9, SD8, SX9, SDB8, BPM, SD9a, Q10, SD8, SX10, SDC4, FCOR, SDC5, BPM, SDC02, BPM, SDC6, FCOR, SDC7, SX10, SD8, Q10, SD91, BPM, SD42, SX9, SD8, Q9, SD7, BEND, SD7, Q6, SD5, SX7, SD41, BPM, SDB9, Q7, SD3, SX8, SD2, Q8, SDC8, FCOR, SDC9, BPM, SDM03, BPM, DRFT10, FCOR, DRFT11, Q8, SD2, SX8, SD3, Q7, SDB10, BPM, SD42, SX7, SD5, Q6, SD7, BEND, SD7, Q9, SD8, SX9, SDB11, BPM, SD92, Q10, SD8, SX10, DRFT12, FCOR, DRFT13, BPM, SDC03, BPM, DRFT14, FCOR, DRFT15, SX10, SD8, Q10, SD93, BPM, SDB12, SX9, SD8, Q9, SD7, BEND, SD7, Q6, SD5, SX7, SD41, BPM, SDB13, Q7, SD3, SX8, SD2, Q8, DRFT16, FCOR, DRFT17, BPM, SDM04, BPM, DRFT18, FCOR, DRFT19, Q8, SD2, SX6, SD3, Q7, SDB14, BPM, SD43, SX5, SD5, Q6, SD7, BEND, SD7, Q4, SD8, SX3, SDB15, BPM, SD9a, Q5, SD12, SX4, SDB4, BPM, SD10a, Q5, SD93, BPM, SDB17, SX3, SD8, Q4, SD7, BEND, SD6, SX2, SD5, Q3, SD141, BPM, SDB18, Q2, SD3, SX1, SD2, Q1, SDC20, FCOR, SDC21, BPM]
    SUP2 = [BPM, SDC2, FCOR, SDC1, Q1, SD2, SX1, SD3, Q2, SDB7, BPM, SD14a, Q3, SD5, SX2, SD6, BEND, SD7, Q4, SD8, SX3, SDB3, BPM, SD9a, Q5, SD12, SX4, SDB4, BPM, SD10a, Q5, SD93, BPM, SDB5, SX3, SD8, Q4, SD7, PXBPM, BEND, SD7, Q6, SD5, SX5, SD41, BPM, SDB6, Q7, SD3, SX6, SD2, Q8, SDC1, FCOR, SDC2, BPM, SDM06, BPM, SDC2, FCOR, SDC1, Q8, SD2, SX8, SD3, Q7, SDB7, BPM, SDB5, SX7, SD5, Q6, SD7, BEND, SD7, Q9, SD8, SX9, SDB3, BPM, SD9a, Q10, SD8, SX10, SDC4, FCOR, SDC5, BPM, SDC06, BPM, DRFT14, FCOR, DRFT15, SX10, SD8, Q10, SD93, BPM, SDB12, SX9, SD8, Q9, SD7, BEND, SD7, Q6, SD5, SX7, SD41, BPM, SDB6, Q7, SD3, SX8, SD2, Q8, SDC1, FCOR, SDC2, BPM, SDM07, BPM, SDC2, FCOR, SDC1, Q8, SD2, SX8, SD3, Q7, SDB7, BPM, SDB5, SX7, SD5, Q6, SD7, BEND, SD7, Q9, SD8, SX9, SDB3, BPM, SD9a, Q10, SD8, SX10, SDC4, FCOR, SDC5, BPM, SDC07, BPM, DRFT14, FCOR, DRFT15, SX10, SD8, Q10, SD93, BPM, SDB12, SX9, SD8, Q9, SD7, BEND, SD7, Q6, SD5, SX7, SD41, BPM, SDB6, Q7, SD3, SX8, SD2, Q8, SDC1, FCOR, SDC2, BPM, SDM08, BPM, SDC2, FCOR, SDC1, Q8, SD2, SX6, SD3, Q7, SDB7, BPM, SD42, SX5, SD5, Q6, SD7, BEND, SD7, Q4, SD8, SX3, SDB3, BPM, SD9a, Q5, SD12, SX4, SDB4, BPM, SD10a, Q5, SD93, BPM, SD42, SX3, SD8, Q4, SD7, BEND, SD6, SX2, SD5, Q3, SD141, BPM, SDB18, Q2, SD3, SX1, SD2, Q1, SDC1, FCOR, SDC2, BPM]
    SUP3 = [BPM, SDC2, FCOR, SDC1, Q1, SD2, SX1, SD3, Q2, SDB7, BPM, SD14a, Q3, SD5, SX2, SD6_BPM_double, BEND, SD7, Q4, SD8, SX3, SDB3, BPM, SD9a, Q5, SD12, SX4, SDB4, BPM, SD10a, Q5, SD93, BPM, SDB5, SX3, SD8, Q4, SD7, PXBPM, BEND, SD7, Q6, SD5, SX5, SD41, BPM, SDB6, Q7, SD3, SX6, SD2, Q8, SDC1, FCOR, SDC2, BPM, SDM10, BPM, SDC2, FCOR, SDC1, Q8, SD2, SX8, SD3, Q7, SDB7, BPM, SDB5, SX7, SD5, Q6, SD7, BEND, SD7, Q9, SD8, SX9, SDB3, BPM, SD9a, Q10, SD8, SX10, SDC4, FCOR, SDC5, BPM, SDC10, BPM, DRFT14, FCOR, DRFT15, SX10, SD8, Q10, SD93, BPM, SDB12, SX9, SD8, Q9, SD7, BEND, SD7, Q6, SD5, SX7, SD41, BPM, SDB6, Q7, SD3, SX8, SD2, Q8, SDC1, FCOR, SDC2, BPM, SDM11, BPM, SDC2, FCOR, SDC1, Q8, SD2, SX8, SD3, Q7, SDB7, BPM, SDB5, SX7, SD5, Q6, SD7, BEND, SD7, Q9, SD8, SX9, SDB3, BPM, SD9a, Q10, SD8, SX10, SDC4, FCOR, SDC5, BPM, SDC11, BPM, DRFT14, FCOR, DRFT15, SX10, SD8, Q10, SD93, BPM, SDB12, SX9, SD8, Q9, SD7, BEND, SD7, Q6, SD5, SX7, SD41, BPM, SDB6, Q7, SD3, SX8, SD2, Q8, SDC1, FCOR, SDC2, BPM, SDM12, BPM, SDC2, FCOR, SDC1, Q8, SD2, SX6, SD3, Q7, SDB7, BPM, SD42, SX5, SD5, Q6, SD7, BEND, SD7, Q4, SD8, SX3, SDB3, BPM, SD9a, Q5, SD12, SX4, SDB4, BPM, SD10a, Q5, SD93, BPM, SD42, SX3, SD8, Q4, SD7, mROCK, ROCK, mROCK, SD6, SX2, SD5, Q3, SD141, BPM, SDB18, Q2, SD3, SX11, SD2, Q1, SDC1, FCOR, SDC2, BPM]
    SUP4 = [BPM, SDC2, FCOR, SDC1, Q1, SD2, SX11, SD3, Q2, SDB7, BPM, SD14a, Q3, SD5, SX2, SD6, BEND, SD7, Q4, SD8, SX3, SDB3, BPM, SD9a, Q5, SD12, SX4, SDB4, BPM, SD10a, Q5, SD93, BPM, SDB5, SX3, SD8, Q4, SD7, PXBPM, BEND, SD7, Q6, SD5, SX5, SD41, BPM, SDB6, Q7, SD3, SX6, SD2, Q8, SDC1, FCOR, SDC2, BPM, SDM14, BPM, SDC2, FCOR, SDC1, Q8, SD2, SX8, SD3, Q7, SDB7, BPM, SDB5, SX7, SD5, Q6, SD7, BEND, SD7, Q9, SD8, SX9, SDB3, BPM, SD9a, Q10, SD8, SX10, SDC4, FCOR, SDC5, BPM, SDC14, BPM, DRFT14, FCOR, DRFT15, SX10, SD8, Q10, SD93, BPM, SDB12, SX9, SD8, Q9, SD7, BEND, SD7, Q6, SD5, SX7, SD41, BPM, SDB6, Q7, SD3, SX8, SD2, Q8, SDC1, FCOR, SDC2, BPM, SDM15, BPM, SDC2, FCOR, SDC1, Q8, SD2, SX8, SD3, Q7, SDB7, BPM, SDB5, SX7, SD5, Q6, SD7, BEND, SD7, Q9, SD8, SX9, SDB3, BPM, SD9a, Q10, SD8, SX10, SDC4, FCOR, SDC5, BPM, SDC15, BPM, DRFT14, FCOR, DRFT15, SX10, SD8, Q10, SD93, BPM, SDB12, SX9, SD8, Q9, SD7, BEND, SD7, Q6, SD5, SX7, SD41, BPM, SDB6, Q7, SD3, SX8, SD2, Q8, SDC1, FCOR, SDC2, BPM, SDM16, BPM, SDC2, FCOR, SDC1, Q8, SD2, SX6, SD3, Q7, SDB7, BPM, SD42, SX5, SD5, Q6, SD7, BEND, SD7, Q4, SD8, SX3, SDB3, BPM, SD9a, Q5, SD12u2, HSCRAP, SDHSCRAP2, HSCRAP, SD12d2, SX4, SDB4, BPM, SD10a, Q5, SD93, BPM, SD42, SX3, SD8, Q4, SD7, BEND, SD6, SX2, SD5, Q3, SD141, BPM, SDB18, Q2, SD3, SX1, SD2, Q1, SDC22, BPM]

    ELIST = [DEBUT, SECT1, SDL01d, SUP1, SECT2, SDL05, SUP2, SECT3, SDL09, SUP3, SECT4, SDL13, SUP4, SDL01u, FIN]
    ele_list = []
    append_ele(ele_list, ELIST)
    ring = ssr.CSLattice(ele_list)
    # ele_idx = get_ele_index(ring.elements)
    ele_idx = ssr.element_index(ring)
    ring = setNanoscopium(ring, ele_idx)
    #
    # % set WSV50 PSICHE PSICHE optics
    ring = setPSICHEoptics(ring, ele_idx)
    #
    # % set Superbend compensation
    ring = setSuperBEND(ring, ele_idx)
    #
    # %% set NANOSCOPIUM tuner magnets
    # ring{ATIndexList.NANOC(1)}.KickAngle(1) =  -5.00e-6*0; % rad
    # ring{ATIndexList.NANOC(2)}.KickAngle(1) =   2.25e-6*0; % rad
    # ring{ATIndexList.NANOC(3)}.KickAngle(1) =  -1.25e-6*0; % rad
    # ring{ATIndexList.NANOC(4)}.KickAngle(1) =  -2.25e-6*0; % rad
    # ring = set_kick_angle(ring, ele_idx['NANOC'][0], -5.00e-6*0)
    ring.linear_optics()
    k1s = ssr.adjust_tunes(ring, quadrupole_name_list=['Q7', 'Q9', 'Q7_4', 'Q9_3', 'Q9_4', 'Q7_5'],
                           target=[v[10], v[11]], initialize=False)
    k2s = ssr.chromaticity_correction(ring, ['S9', 'S10'], target=[1.24, 2.34])
    return ring, k2s, k1s


def setNanoscopium(ring, ATIndexList):
    QP1N = -1.336224e+00
    QP2N = 1.874242e+00
    QP3N = -1.126772e+00
    ring = setquad(ring, ATIndexList['Q1'][6-1], QP1N, 6)
    ring = setquad(ring, ATIndexList['Q1'][7-1], QP1N, 7)
    ring = setquad(ring, ATIndexList['Q2'][6-1], QP2N, 6)
    ring = setquad(ring, ATIndexList['Q2'][7-1], QP2N, 7)
    ring = setquad(ring, ATIndexList['Q3'][6-1], QP3N, 6)
    ring = setquad(ring, ATIndexList['Q3'][7-1], QP3N, 7)

    QI1 = -1.148605e+00
    QI2 = 1.698188e+00
    QI3 = -8.977849e-01
    QI4 = -1.032541e+00
    QI5 = 1.776718e+00
    QI51 = 1.551736e+00
    ring = setquad(ring, ATIndexList['Q1'][1-1], QI1, 1)
    ring = setquad(ring, ATIndexList['Q1'][8-1], QI1, 8)
    ring = setquad(ring, ATIndexList['Q1'][4-1], QI1, 4)
    ring = setquad(ring, ATIndexList['Q1'][5-1], QI1, 5)

    ring = setquad(ring, ATIndexList['Q2'][1-1], QI2, 1)
    ring = setquad(ring, ATIndexList['Q2'][8-1], QI2, 8)
    ring = setquad(ring, ATIndexList['Q2'][4-1], QI2, 4)
    ring = setquad(ring, ATIndexList['Q2'][5-1], QI2, 5)

    ring = setquad(ring, ATIndexList['Q3'][1-1], QI3, 1)
    ring = setquad(ring, ATIndexList['Q3'][8-1], QI3, 8)
    ring = setquad(ring, ATIndexList['Q3'][4-1], QI3, 4)
    ring = setquad(ring, ATIndexList['Q3'][5-1], QI3, 5)

    ring = setquad(ring, ATIndexList['Q4'][1-1], QI4, 1)
    ring = setquad(ring, ATIndexList['Q4'][16-1], QI4, 16)
    ring = setquad(ring, ATIndexList['Q4'][8-1], QI4, 8)
    ring = setquad(ring, ATIndexList['Q4'][9-1], QI4, 9)

    ring = setquad(ring, ATIndexList['Q5'][1-1], QI5, 1)
    ring = setquad(ring, ATIndexList['Q5'][16-1], QI5, 16)
    ring = setquad(ring, ATIndexList['Q5'][8-1], QI5, 8)
    ring = setquad(ring, ATIndexList['Q5'][9-1], QI5, 9)

    ring = setquad(ring, ATIndexList['Q5'][2-1], QI51, 2)
    ring = setquad(ring, ATIndexList['Q5'][15-1], QI51, 15)
    ring = setquad(ring, ATIndexList['Q5'][7-1], QI51, 7)
    ring = setquad(ring, ATIndexList['Q5'][10-1], QI51, 10)
    return ring


def setPSICHEoptics(ring, ATIndexList):
    # % Quadrupoles modifies pour l'optique PSICHE

    QW81 = -1.382486
    QW8  = -1.110309
    QW7  =  1.916854
    QW6  = -1.086326
    QW9  = -1.593204
    QW10 =  1.769764
    ring = setquad(ring, ATIndexList['Q8'][3-1], QW81, 3)
    ring = setquad(ring, ATIndexList['Q8'][6-1], QW81, 6)
    ring = setquad(ring, ATIndexList['Q8'][4-1], QW8, 4)
    ring = setquad(ring, ATIndexList['Q8'][5-1], QW8, 5)
    ring = setquad(ring, ATIndexList['Q7'][4-1], QW7, 4)
    ring = setquad(ring, ATIndexList['Q7'][5-1], QW7, 5)
    ring = setquad(ring, ATIndexList['Q6'][4-1], QW6, 4)
    ring = setquad(ring, ATIndexList['Q6'][5-1], QW6, 5)
    ring = setquad(ring, ATIndexList['Q9'][3-1], QW9, 3)
    ring = setquad(ring, ATIndexList['Q9'][4-1], QW9, 4)
    ring = setquad(ring, ATIndexList['Q10'][3-1], QW10, 3)
    ring = setquad(ring, ATIndexList['Q10'][4-1], QW10, 4)
    return ring


def setSuperBEND(ring, ATIndexList):
    """% set Superbend compensation

    % Quadrupoles modifies pour compenser la focalisation du Superbend ROCK"""
    a1 = 1 - 1.5975e-2
    a2 = 1 - 0.0913e-2
    a3 = 1 + 0.3239e-2
    a41 = 1 + 0.6104e-2
    a42 = 1 - 6.9034e-2
    a51 = 1 + 0.1668e-2
    a52 = 1 - 1.8716e-2

    QP1Nrock = -1.336224e+00 * a1
    QP2Nrock = 1.874242e+00 * a2
    QP3Nrock = -1.126772e+00 * a3
    Q41rock = -1.229389e+00 * a41
    Q42rock = -1.229389e+00 * a42
    Q51rock = 1.713696e+00 * a51
    Q52rock = 1.713696e+00 * a52

    ring = setquad(ring, ATIndexList['Q1'][6-1], QP1Nrock, 6)
    ring = setquad(ring, ATIndexList['Q2'][6-1], QP2Nrock, 6)
    ring = setquad(ring, ATIndexList['Q3'][6-1], QP3Nrock, 6)
    ring = setquad(ring, ATIndexList['Q4'][11-1], Q41rock, 11)
    ring = setquad(ring, ATIndexList['Q4'][12-1], Q42rock, 12)
    ring = setquad(ring, ATIndexList['Q5'][11-1], Q51rock, 11)
    ring = setquad(ring, ATIndexList['Q5'][12-1], Q52rock, 12)
    return ring


def setquad(ring, idx, k, idx_in_family):
    ring.elements[idx].k1 = k
    ring.elements[idx].name += f'_{idx_in_family}'
    return ring

# def get_ele_index(ele_list):
#     ele_idx = {}
#     for i, ele in enumerate(ele_list):
#         if ele.name not in ele_idx:
#             ele_idx[ele.name] = [i]
#         else:
#             ele_idx[ele.name].append(i)
#     return ele_idx


def append_ele(elelist=None, matlab_list=None):
    for ele in matlab_list:
        if isinstance(ele, list):
            append_ele(elelist, ele)
        else:
            elelist.append(ele)
