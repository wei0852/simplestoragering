from .components import LineEnd
from .rfcavity import RFCavity
from .particles import RefParticle, Beam7
from .constants import pi, c, Cl, Cr, LENGTH_PRECISION
import numpy as np
from copy import deepcopy


class SlimRing(object):
    """lattice object, solve by slim method"""

    def __init__(self, ele_list: list):
        self.length = 0
        self.elements = []
        for ele in ele_list:
            # ele.s = self.length
            self.elements.append(ele)
            self.length = round(self.length + ele.length, LENGTH_PRECISION)
        self.rf_cavity = None
        self.ele_slices = None
        self.damping = None
        self.U0 = 0
        self.f_c = 0
        self.__set_rf()
        self.__slice()
        self.solve_closed_orbit()
        self.solve_damping()
        # self.along_ring()

    def __set_rf(self):
        """solve U0 and set rf parameters"""
        i2 = 0
        for ele in self.elements:
            i2 = i2 + ele.length * ele.h ** 2
            if isinstance(ele, RFCavity):
                self.rf_cavity = ele
        self.U0 = Cr * RefParticle.energy ** 4 * i2 / (2 * pi)
        # self.T_period = self.length * self.periods_number / (c * RefParticle.beta)
        self.f_c = c * RefParticle.beta / self.length
        if self.rf_cavity is not None:
            self.rf_cavity.f_c = self.f_c

    def __slice(self):
        self.ele_slices = []
        current_s = 0
        current_identifier = 0
        for ele in self.elements:
            [new_list, current_s] = ele.slice(current_s, current_identifier)
            self.ele_slices += new_list
            current_identifier += 1
        last_ele = LineEnd(s=self.length, identifier=current_identifier)
        self.ele_slices.append(last_ele)

    def solve_closed_orbit(self):
        """solve closed orbit, iterate to solve, renew the x0 and matrix"""
        x0 = np.array([0, 0, 0, 0, 0, 0])
        matrix7 = np.identity(7)
        i = 1
        print('\n-------------------\nsearching closed orbit:')
        while i < 300:
            for ele in self.ele_slices:
                ele.closed_orbit = deepcopy(x0)
                x0 = ele.next_closed_orbit
                matrix7 = ele.closed_orbit_matrix.dot(matrix7)
            coefficient_matrix = (matrix7 - np.identity(7))[0: 6, 0: 6]
            result_vec = -matrix7[0: 6, 6].T
            x0 = np.linalg.solve(coefficient_matrix, result_vec)
            print(f'\niterated {i} times, current result is: \n    {x0}')  # TODO: this is not iteration
            i += 1
            if max(abs(x0 - self.ele_slices[0].closed_orbit)) < 1e-8:
                break
        print(f'\nclosed orbit at s=0 is \n    {x0}\n--------------')

    def track_close_orbit(self):
        print('\n-------------------\ntracking closed orbit:\n')
        xco = np.zeros(6)
        matrix = np.zeros([6, 6])
        resdl = 1
        j = 1
        beam = Beam7(xco)
        while j <= 10 and resdl > 1e-16:
            beam = Beam7(xco)
            for ele in self.ele_slices:
                ele.closed_orbit = np.array(beam.get_particle_array())
                beam = ele.real_track(beam)
            for i in range(6):
                matrix[:, i] = (beam.matrix[:, i] - beam.matrix[:, 6]) / beam.precision
            d = beam.matrix[:, 6] - xco
            dco = np.linalg.inv(np.identity(6) - matrix).dot(d)
            xco = xco + dco
            resdl = dco.dot(dco.T)
            print(f'iterated {j} times, current result is \n    {beam.matrix[:, 6]}\n')
            j += 1
        print(f'closed orbit at s=0 is \n    {xco}\n')
        # print(f'{matrix}\n')
        eig_val, ring_eig_matrix = np.linalg.eig(matrix)
        self.damping = - np.log(np.abs(eig_val))
        print(f'damping  = {self.damping}')
        print(f'damping time = {1 / self.f_c / self.damping}')
        print('\ncheck:')
        print(f'sum damping = {self.damping[0] + self.damping[2] + self.damping[4]}, '
              f'2U0/E0 = {2 * self.U0 / RefParticle.energy}')
        print(f'\nring tune = {np.angle(eig_val) / 2 / pi}')
        print('\n--------------------------------------------\n')
        # solve coefficient
        beam = Beam7(xco)
        ave_deco_square = np.zeros(6)
        sideways_photons = np.zeros(6)
        for ele in self.ele_slices:
            beam = ele.real_track(beam)
            matrix = beam.solve_transfer_matrix()
            eig_matrix = matrix.dot(ring_eig_matrix)
            if ele.h != 0:
                for k in range(6):
                    ave_deco_square[k] += abs(eig_matrix[4, k]) ** 2 * abs(ele.h) ** 3 * ele.length
                    sideways_photons[k] += abs(eig_matrix[2, k]) ** 2 * abs(ele.h) ** 3 * ele.length
        for k in range(6):
            ave_deco_square[k] = ave_deco_square[k] * Cl * RefParticle.gamma ** 5 / c / self.damping[k]
            sideways_photons[k] = sideways_photons[k] * Cl * RefParticle.gamma ** 3 / c / self.damping[k] / 2
        # solve equilibrium beam
        eig_matrix = ring_eig_matrix
        beam = Beam7(xco)
        for ele in self.ele_slices:
            equilibrium_beam = np.zeros((6, 6))
            for j in range(6):
                for i in range(j + 1):
                    for k in range(6):
                        equilibrium_beam[i, j] += ((ave_deco_square[k] + sideways_photons[k]) *
                                                   np.real(eig_matrix[i, k] * np.conj(eig_matrix[j, k])))
            for i in range(6):
                for j in range(i):
                    equilibrium_beam[i, j] = equilibrium_beam[j, i]
            ele.beam = deepcopy(equilibrium_beam)
            beam = ele.real_track(beam)
            eig_matrix = beam.solve_transfer_matrix().dot(ring_eig_matrix)

    def solve_damping(self):
        matrix = np.identity(6)
        for ele in self.ele_slices:
            matrix = ele.damping_matrix.dot(matrix)
        eig_val, eig_matrix = np.linalg.eig(matrix)
        self.damping = - np.log(np.abs(eig_val))
        print(f'damping  = {self.damping}')
        print(f'damping time = {1 / self.f_c / self.damping}')
        print('\ncheck:')
        print(f'sum damping = {self.damping[0] + self.damping[2] + self.damping[4]}, '
              f'2U0/E0 = {2 * self.U0 / RefParticle.energy}')
        print('\n--------------------------------------------\n')

    def along_ring(self):
        """solve tune along the ring. use matrix"""
        matrix = np.identity(6)
        for ele in self.ele_slices:
            matrix = ele.matrix.dot(matrix)
        eig_val, ring_eig_matrix = np.linalg.eig(matrix)  # Ei is eig_matrix[:, i]  E_ki is eig_matrix[i, k]
        print(f'ring tune = {np.angle(eig_val) / 2 / pi}\n')
        # solve average decomposition and tune along the lattice
        ave_deco_square = np.zeros(6)
        sideways_photons = np.zeros(6)
        eig_matrix = ring_eig_matrix
        for ele in self.ele_slices:
            eig_matrix = ele.matrix.dot(eig_matrix)
            if ele.h != 0:
                for k in range(6):
                    ave_deco_square[k] += abs(eig_matrix[4, k]) ** 2 * abs(ele.h) ** 3 * ele.length
                    sideways_photons[k] += abs(eig_matrix[2, k]) ** 2 * abs(ele.h) ** 3 * ele.length
        for k in range(6):
            ave_deco_square[k] = ave_deco_square[k] * Cl * RefParticle.gamma ** 5 / c / self.damping[k]
            sideways_photons[k] = sideways_photons[k] * Cl * RefParticle.gamma ** 3 / c / self.damping[k] / 2
        # solve equilibrium beam
        eig_matrix = ring_eig_matrix
        for ele in self.ele_slices:
            equilibrium_beam = np.zeros((6, 6))
            for j in range(6):
                for i in range(j + 1):
                    for k in range(6):
                        equilibrium_beam[i, j] += ((ave_deco_square[k] + sideways_photons[k]) *
                                                   np.real(eig_matrix[i, k] * np.conj(eig_matrix[j, k])))
            for i in range(6):
                for j in range(i):
                    equilibrium_beam[i, j] = equilibrium_beam[j, i]
            ele.beam = deepcopy(equilibrium_beam)
            eig_matrix = ele.matrix.dot(eig_matrix)

    def along_ring_damping_matrix(self):
        """solve tune along the ring. use matrix"""
        matrix = np.identity(6)
        for ele in self.ele_slices:
            matrix = ele.damping_matrix.dot(matrix)
        eig_val, ring_eig_matrix = np.linalg.eig(matrix)  # Ei is eig_matrix[:, i]  E_ki is eig_matrix[i, k]
        print(f'ring tune = {np.angle(eig_val) / 2 / pi}\n')
        # solve average decomposition and tune along the lattice
        ave_deco_square = np.zeros(6)
        sideways_photons = np.zeros(6)
        eig_matrix = ring_eig_matrix
        for ele in self.ele_slices:
            eig_matrix = ele.damping_matrix.dot(eig_matrix)
            if ele.h != 0:
                for k in range(6):
                    ave_deco_square[k] += abs(eig_matrix[4, k]) ** 2 * abs(ele.h) ** 3 * ele.length
                    sideways_photons[k] += abs(eig_matrix[2, k]) ** 2 * abs(ele.h) ** 3 * ele.length
        for k in range(6):
            ave_deco_square[k] = ave_deco_square[k] * Cl * RefParticle.gamma ** 5 / c / self.damping[k]
            sideways_photons[k] = sideways_photons[k] * Cl * RefParticle.gamma ** 3 / c / self.damping[k] / 2
        # solve equilibrium beam
        eig_matrix = ring_eig_matrix
        for ele in self.ele_slices:
            equilibrium_beam = np.zeros((6, 6))
            for j in range(6):
                for i in range(j + 1):
                    for k in range(6):
                        equilibrium_beam[i, j] += ((ave_deco_square[k] + sideways_photons[k]) *
                                                   np.real(eig_matrix[i, k] * np.conj(eig_matrix[j, k])))
            for i in range(6):
                for j in range(i):
                    equilibrium_beam[i, j] = equilibrium_beam[j, i]
            ele.beam = deepcopy(equilibrium_beam)
            eig_matrix = ele.damping_matrix.dot(eig_matrix)

    def matrix_output(self, file_name: str = 'matrix.txt'):
        """output uncoupled matrix for each element and contained matrix"""

        matrix = np.identity(6)
        file = open(file_name, 'w')
        location = 0.0
        for ele in self.elements:
            file.write(f'{ele.type()} {ele.name} at s={location},  {ele.magnets_data()}\n')
            location = round(location + ele.length, LENGTH_PRECISION)
            file.write(str(ele.matrix) + '\n')
            file.write('contained matrix:\n')
            matrix = ele.matrix.dot(matrix)
            file.write(str(matrix))
            file.write('\n\n--------------------------\n\n')
        file.close()

    def coupled_matrix_output(self, filename: str = 'matrix.txt'):
        matrix = np.identity(6)
        element_matrix = np.identity(6)
        file = open(filename, 'w')
        location = 0.0
        first_ele = self.ele_slices[0]
        last_identifier = first_ele.identifier
        file.write(f'{first_ele.type()} {first_ele.name} at s={location} \n')
        file.write(f'closed orbit: \n    {first_ele.closed_orbit}\n')
        for ele in self.ele_slices:
            if ele.identifier != last_identifier:
                matrix = element_matrix.dot(matrix)
                file.write('element matrix:\n' + str(element_matrix) + '\n')
                file.write('contained matrix:\n')
                file.write(str(matrix))
                element_matrix = np.identity(6)
                file.write('\n\n--------------------------------------------\n\n')
                file.write(f'{ele.type()} {ele.name} at s={location} \n')
                file.write(f'closed orbit: {ele.closed_orbit}\n')
            element_matrix = ele.matrix.dot(element_matrix)
            location = round(location + ele.length, LENGTH_PRECISION)
            last_identifier = ele.identifier
        matrix = element_matrix.dot(matrix)
        file.write(str(element_matrix) + '\n')
        file.write('full matrix:\n')
        file.write(str(matrix))
        file.close()

    def output_equilibrium_beam(self, filename: str = 'equilibrium_beam.txt'):
        file = open(filename, 'w')
        location = 0.0
        last_identifier = 123465
        for ele in self.ele_slices:
            if ele.identifier != last_identifier:
                file.write(f'{ele.type()} {ele.name} at s={location} \n')
                file.write(f'closed orbit: {ele.closed_orbit}\n')
                file.write('equilibrium beam:\n')
                file.write(str(ele.beam))
                file.write('\n\n--------------------------------------------\n\n')
            location = round(location + ele.length, LENGTH_PRECISION)
            last_identifier = ele.identifier
        file.close()


def compute_twiss_of_slim_method(slim_ring: SlimRing):
    """compute twiss parameters according to equilibrium beam matrix"""

    betax_list = []
    etax_list = []
    betay_list = []
    for ele in slim_ring.ele_slices:
        betax, betay, etax = __compute_twiss_from_beam_matrix(ele.beam)
        betax_list.append(betax)
        betay_list.append(betay)
        etax_list.append(etax)
    return betax_list, betay_list, etax_list


def __compute_twiss_from_beam_matrix(current_beam):
    etax = current_beam[0, 5] / current_beam[5, 5]
    etaxp = current_beam[1, 5] / current_beam[5, 5]
    sigma_11_beta = current_beam[0, 0] - current_beam[5, 5] * etax ** 2
    sigma_22_beta = current_beam[1, 1] - current_beam[5, 5] * etaxp ** 2
    sigma_12_beta = current_beam[0, 1] - current_beam[5, 5] * etaxp * etax
    emitt_x_beta = np.sqrt(sigma_11_beta * sigma_22_beta - sigma_12_beta ** 2)
    betax = sigma_11_beta / emitt_x_beta
    # vertical
    emitt_y = np.sqrt(current_beam[2, 2] * current_beam[3, 3] - current_beam[2, 3] ** 2)
    betay = current_beam[2, 2] / emitt_y
    return betax, betay, etax
