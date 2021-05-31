import matplotlib.pyplot as plt
import numpy as np

from main import *
import time

qua_slice = 23

Particle.set_energy(800.)
DC1 = Drift('DC1', 1.943675)
BS1 = Drift('BS1', 0.060040)
DC2 = Drift('DC2', 0.116855)
DC3 = Drift('DC3', 0.257690)
DC4 = Drift('DC4', 0.198000)
BQ2 = Drift('BQ2', 0.093315)
DC5 = Drift('DC5', 0.605000)
BQ1 = Drift('BQ1', 0.086390)
DC6 = Drift('DC6', 1.103675)
DQS1 = Drift('DQS1', 0.107690)
DQS2 = Drift('DQS2', 0.107765)

Q01 = Quadrupole('Q01', 0.217370, 3.5895, n_slices=qua_slice)
Q02 = Quadrupole('Q02', 0.217370, -2.9627, n_slices=qua_slice)
Q03 = Quadrupole('Q03', 0.317220, 3.4244, n_slices=qua_slice)
Q04 = Quadrupole('Q04', 0.217370, -1.8621, n_slices=qua_slice)
Q05 = Quadrupole('Q05', 0.217370, -2.0783, n_slices=qua_slice)
Q06 = Quadrupole('Q06', 0.317220, 2.7206, n_slices=qua_slice)
Q07 = Quadrupole('Q07', 0.217370, -2.7852, n_slices=qua_slice)
Q08 = Quadrupole('Q08', 0.217370, 3.6109, n_slices=qua_slice)

S1 = Sextupole('S1', 0.1240, 8.006)
S2 = Sextupole('S2', 0.1172, -12.738)
S3 = Sextupole('S3', 0.1172, 50.516)
S4 = Sextupole('S4', 0.1240, -74.214)

BEND = HBend('BEND', 1.7, 0.7853981634, 0.3926990817, 0.3926990817, n_slices=170)

rf_ca = RFCavity('rf', voltage_in_MeV=0.26, frequency=204e6, phase=3.077164602543054)

segment1 = [DC1, BS1, S1, DC2, Q01, DC3, S2, DQS1, Q02, BQ2, DC4, BEND, DC5, BQ1, Q03, DQS2, S3, DC3, Q04,
            DC2, S4, BS1, DC6, DC6, BS1, S4, DC2, Q05, DC3, S3, DQS2, Q06, BQ1, DC5, BEND, DC4, BQ2, Q07,
            DQS1, S2, DC3, Q08, DC2, S1, BS1, DC1]
segment2 = [DC1, BS1, S1, DC2, Q08, DC3, S2, DQS1, Q07, BQ2, DC4, BEND, DC5, BQ1, Q06, DQS2, S3, DC3, Q05,
            DC2, S4, BS1, DC6, DC6, BS1, S4, DC2, Q04, DC3, S3, DQS2, Q03, BQ1, DC5, BEND, DC4, BQ2, Q02,
            DQS1, S2, DC3, Q01, DC2, S1, BS1, DC1]
segment3 = [DC1, BS1, S1, DC2, Q01, DC3, S2, DQS1, Q02, BQ2, DC4, BEND, DC5, BQ1, Q03, DQS2, S3, DC3, Q04,
            DC2, S4, BS1, DC6, DC6, BS1, S4, DC2, Q05, DC3, S3, DQS2, Q06, BQ1, DC5, BEND, DC4, BQ2, Q07,
            DQS1, S2, DC3, Q08, DC2, S1, BS1, DC1]
segment4 = [DC1, BS1, S1, DC2, Q08, DC3, S2, DQS1, Q07, BQ2, DC4, BEND, DC5, BQ1, Q06, DQS2, S3, DC3, Q05,
            DC2, S4, BS1, DC6, rf_ca, DC6, BS1, S4, DC2, Q04, DC3, S3, DQS2, Q03, BQ1, DC5, BEND, DC4, BQ2, Q02,
            DQS1, S2, DC3, Q01, DC2, S1, BS1, DC1]
segment4i = [DC1, BS1, S1, DC2, Q08, DC3, S2, DQS1, Q07, BQ2, DC4, BEND, DC5, BQ1, Q06, DQS2, S3, DC3, Q05,
             DC2, S4, BS1, DC6,  DC6, BS1, S4, DC2, Q04, DC3, S3, DQS2, Q03, BQ1, DC5, BEND, DC4, BQ2, Q02,
             DQS1, S2, DC3, Q01, DC2, S1, BS1, DC1]

# start = time.time()
np.set_printoptions(precision=8, suppress=True, linewidth=100)

slim_lattice = SlimRing(segment1 + segment2 + segment3 + segment4)
# end = time.time()
print(rf_ca.voltage * np.sin(rf_ca.phase))
# print(2 * rf_ca.voltage * np.sin(rf_ca.phase) / Particle.energy)
slim_lattice.track_close_orbit()
# slim_lattice.output_equilibrium_beam()
# print(f'\nslim run time: {end - start}')
# start = time.time()
# cs_lattice = CSLattice(segment1 + segment2 + segment3 + segment4, 1, 0.01)
# end = time.time()
# print(f'\ncs run time: {end - start}')
# print(cs_lattice)
# print('\n-------------------------------\n')
print(np.sqrt(slim_lattice.ele_slices[0].beam[5, 5]))


# def emmit_x_beta(current_beam):
#     return np.sqrt((current_beam[0, 0] - current_beam[0, 5] ** 2 / ) * (current_beam[1, 1] - current_beam[1, 5])
#                    - (current_beam[0, 1] - current_beam[0, 5] * current_beam[1, 5] / current_beam[5, 5]) ** 2)


# beam = slim_lattice.ele_slices[0].beam
# emmitx = np.sqrt(beam[0, 0] * beam[1, 1] - beam[0, 1] ** 2)
# print(f'emmit x = {emmitx}')
# emmitx_beta = np.sqrt((beam[0, 0] - beam[0, 5] ** 2 / beam[5, 5]) * (beam[1, 1] - beam[1, 5] ** 2 / beam[5, 5])
#                       - (beam[0, 1] - beam[0, 5] * beam[1, 5] / beam[5, 5]) ** 2)
# # emmity = np.sqrt(beam[2, 2] * beam[3, 3] - beam[2, 3] ** 2)
# emmity_beta = np.sqrt((beam[2, 2] - beam[2, 5] ** 2 / beam[5, 5]) * (beam[3, 3] - beam[3, 5] ** 2 / beam[5, 5])
#                       - (beam[2, 3] - beam[2, 5] * beam[3, 5] / beam[5, 5]) ** 2)
# print(f'emmit x beta = {emmitx_beta}')
# print(f'emmit xy = {emmity * emmitx}')
# # print(np.sqrt(beam[4, 4] * beam[5, 5] - beam[4, 5] ** 2))
# slim_eta = []
# slim_betax = []
# sigma11 = []
# sigma33 = []
# s = []
# for ele in slim_lattice.ele_slices:
#     slim_eta.append(ele.beam[0, 5] / ele.beam[5, 5])
#     slim_betax.append((ele.beam[0, 0] - ele.beam[0, 5]) / emmitx_beta)
#     sigma11.append(ele.beam[0, 0])
#     sigma33.append(ele.beam[2, 2])
#     s.append(ele.s)
# plt.plot(s, slim_betax, label='slim')
# # plt.plot(s, sigma33, label='sigma33')
# betax = []
# etax = []
# s1 = []
# for ele in cs_lattice.ele_slices:
#     betax.append(ele.betax)
#     etax.append(ele.etax)
#     s1.append(ele.s)
# plt.plot(s1, betax, label='cs')
# plt.legend()
# plt.title('betax')
# plt.show()


