# -*- coding: utf-8 -*-
import copy
from abc import ABCMeta, abstractmethod
import random

import matplotlib.pyplot as plt
import numpy as np


def get_random(min_data, max_data):
    """generate an random data in [min, max)"""
    random_data = min_data + random.random()*(max_data - min_data)
    if random_data == max_data:
        random_data = get_random(min_data, max_data)
    return random_data


def swap(front, index1, index2):
    temp = front[index1]
    front[index1] = front[index2]
    front[index2] = temp


class AbstractIndividual(metaclass=ABCMeta):
    """individual, need to rewrite initial() and evaluate()"""

    def __init__(self):
        self.vars = []
        self.constraint = []
        self.objs = []
        self.feasible = True
        self.constraint_violation = None
        self.distance = 0
        self.rank = None
        self.ss = []
        self.ndom = 0

    @abstractmethod
    def evaluate(self):
        """calculate the object and constraint, if feasible, calculate constraint violation."""
        pass

    @abstractmethod
    def get_scatter_data(self):
        """get data to plot, return [x, y] or [x, y, color] or [x, y, color, size].

        Color is a set of data reflected in the color, not necessarily the letter or hexadecimal number of the color """

    @abstractmethod
    def the_first_line(self):
        """print to the first line"""

    def mutate(self, min_vars, max_vars):
        """Routine for real polynomial mutation of an individual"""

        p_mut = 0.2
        ETA_M = 5
        for i in range(len(self.vars)):
            rnd = get_random(0, 1)
            if rnd <= p_mut:
                y = self.vars[i]
                yl = min_vars[i]
                yu = max_vars[i]

                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)

                rnd = get_random(0, 1)
                mut_pow = 1 / (ETA_M + 1)
                if rnd <= 0.5:
                    xy = 1 - delta1
                    val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy, (ETA_M + 1.0)))
                    deltaq = pow(val, mut_pow) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (pow(xy, (ETA_M + 1.0)))
                    deltaq = 1.0 - (pow(val, mut_pow))
                y = y + deltaq * (yu - yl)
                if y < yl:
                    y = yl
                if y > yu:
                    y = yu
                self.vars[i] = y

    def check_dominance(self, another_individual):
        """if this individual is dominant, return 1. If another individual is dominant, return -1. If both are
        non-dominant, return 0 """
        flag1 = 0
        flag2 = 0
        if self.feasible and not another_individual.feasible:
            return 1
        elif not self.feasible and another_individual.feasible:
            return -1
        elif not self.feasible and not another_individual.feasible:
            if self.constraint_violation < another_individual.constraint_violation:
                return 1
            elif self.constraint_violation > another_individual.constraint_violation:
                return -1
            else:
                return 0
        else:
            for i in range(len(self.objs)):
                if self.objs[i] < another_individual.objs[i]:
                    flag1 = 1
                elif self.objs[i] > another_individual.objs[i]:
                    flag2 = 1

        if flag1 == 1 and flag2 == 0:
            return 1
        elif flag1 == 0 and flag2 == 1:
            return -1
        else:
            return 0

    @abstractmethod
    def __copy__(self):
        pass


class Population(object):
    """population"""

    def __init__(self, size: int, max_vars: list, min_vars: list, num_vars: int, num_objs: int):
        self.individuals = []
        self.size = size
        self.p_cross = 0.9
        assert len(max_vars) == len(min_vars) == num_vars
        self.max_vars = max_vars
        self.min_vars = min_vars
        self.num_vars = num_vars
        self.num_objs = num_objs
        self.rank_counter = 0

    def initialize(self,  individual: AbstractIndividual):
        """initialize population with random parameters."""
        self.individuals = []
        individual.vars = []
        for i in range(self.size):
            self.individuals.append(copy.deepcopy(individual))
        for ind in self.individuals:
            for i in range(self.num_vars):
                ind.vars.append(get_random(self.min_vars[i], self.max_vars[i]))

    def evaluate(self):
        """calculate the constraint"""
        for i in self.individuals:
            i.evaluate()

    def non_dominated_sort(self):
        front = []
        for ind in self.individuals:
            ind.ss = []
            ind.ndom = 0
            for ano_ind in self.individuals:
                flag = ind.check_dominance(ano_ind)
                if flag == 1:
                    ind.ss.append(ano_ind)
                elif flag == -1:
                    ind.ndom += 1
            if ind.ndom == 0:
                ind.rank = 1
                front.append(ind)
        self.rank_counter = 1
        self.__calculate_crowding_distance(front)
        while front:
            q = []
            for ind in front:
                for dominated_ind in ind.ss:
                    dominated_ind.ndom -= 1
                    if dominated_ind.ndom == 0:
                        dominated_ind.rank = self.rank_counter + 1
                        q.append(dominated_ind)
            if q:
                self.rank_counter += 1
                front = q
                self.__calculate_crowding_distance(front)
            else:
                break

    def __calculate_crowding_distance(self, front):
        """calculating crowding distance

        the crowding distance is the normalized sum of distance between its nearest neighbors in each division."""
        for ind in front:
            ind.distance = 0
        # the min and max data has inf crowding distance
        if len(front) == 1:
            front[0].distance = np.Inf
            return
        if len(front) == 2:
            front[0].distance = np.Inf
            front[1].distance = np.Inf
            return
        for i in range(self.num_objs):  # each division in objective space
            self.__sort_on_obj(front, i, 0, len(front) - 1)
            front[0].distance = np.inf
            front[-1].distance = np.inf
            obj_range = front[-1].objs[i] - front[0].objs[i]  # obj_range = nan if both are inf.
            for j in range(len(front) - 2):
                if obj_range == 0 or front[-1].objs[i] == np.Inf or front[0].objs[i] == np.Inf:
                    front[j+1].distance = np.inf
                else:
                    next_obj = front[j+2].objs[i]
                    pre_obj = front[j].objs[i]
                    front[j+1].distance += (next_obj - pre_obj) / obj_range
        for ind in front:  # normalize
            if ind.distance != np.inf:
                ind.distance = ind.distance / self.num_objs

    def __sort_on_obj(self, front, obj_count, left, right):
        """"""
        # TODO: 直接对front进行交换还是另外定义一个数组表示front的索引？对运算速度是否会有影响
        flag = True
        if left < right:
            for i in range(right - left - 1):
                flag = flag and (front[left + i].objs[obj_count] == front[left + i + 1].objs[obj_count])
            if flag:
                return
            pivot = self.__rand_partition(front, obj_count, left, right)
            self.__sort_on_obj(front, obj_count, left, pivot - 1)
            self.__sort_on_obj(front, obj_count, pivot + 1, right)

    def __rand_partition(self, front, obj_count, left, right):
        rand_index = int(get_random(left, right))
        swap(front, rand_index, right)
        pivot = front[right].objs[obj_count]
        index = left - 1
        for i in range(right - left):
            if front[left + i].objs[obj_count] < pivot:
                index += 1
                swap(front, index, i + left)
        swap(front, index + 1, right)
        return index + 1

    def next_generation(self):

        def __get_distance(ele):
            return ele.distance

        count = 0
        while count < self.size:
            rand_idx1 = int(get_random(0, self.size))
            rand_idx2 = int(get_random(0, self.size))
            parent_idx2 = parent_idx1 = self.__tournament(self.__tournament(rand_idx1, rand_idx2), int(get_random(0, self.size)))
            while parent_idx1 == parent_idx2:
                rand_idx1 = int(get_random(0, self.size))
                rand_idx2 = int(get_random(0, self.size))
                parent_idx2 = self.__tournament(self.__tournament(rand_idx1, rand_idx2), int(get_random(0, self.size)))
            child1, child2 = self.__sbx_crossover(self.individuals[parent_idx1], self.individuals[parent_idx2])
            child1.mutate(self.min_vars, self.max_vars)
            child2.mutate(self.min_vars, self.max_vars)
            child1.evaluate()
            child2.evaluate()
            self.individuals.append(child1)
            self.individuals.append(child2)
            count = count + 2
        self.non_dominated_sort()

        # generate new population
        new_pop = Population(self.size, self.max_vars, self.min_vars, self.num_vars, self.num_objs)
        front = []
        front_counter = 1
        while True:
            for ind in self.individuals:
                if ind.rank == front_counter:
                    front.append(ind)
            front_counter += 1
            if (len(new_pop.individuals) + len(front)) < new_pop.size:
                new_pop.individuals = new_pop.individuals + front
                front = []
            else:
                front.sort(key=__get_distance, reverse=True)
                for ind in front:
                    new_pop.individuals.append(ind)
                    if len(new_pop.individuals) == new_pop.size:
                        break
                break
        new_pop.rank_counter = front_counter - 1
        return new_pop

    def __tournament(self, idx1, idx2):
        """select a better parent"""
        ind1 = self.individuals[idx1]
        ind2 = self.individuals[idx2]
        if ind1.rank < ind2.rank:
            return idx1
        elif ind1.rank > ind2.rank:
            return idx2
        elif ind1.distance > ind2.distance:
            return idx1
        elif ind1.distance < ind2.distance:
            return idx2
        elif get_random(0, 1) <= 0.5:
            return idx1
        else:
            return idx2

    def __sbx_crossover(self, parent1, parent2):
        child1 = copy.copy(parent1)
        child2 = copy.copy(parent2)
        if get_random(0, 1) <= self.p_cross:
            for i in range(self.num_vars):
                if get_random(0, 1) < 0.5:
                    if abs(parent1.vars[i] - parent2.vars[i]) > 1e-10:
                        if parent1.vars[i] < parent2.vars[i]:
                            y1 = parent1.vars[i]
                            y2 = parent2.vars[i]
                        else:
                            y1 = parent2.vars[i]
                            y2 = parent1.vars[i]
                        yl = self.min_vars[i]
                        yu = self.max_vars[i]

                        eta_c = 10

                        rand_data = get_random(0, 1)
                        # beta = 1 + (2 * (y1 - yl) / (y2 - y1))
                        # alpha = 2 - pow(beta, - (eta_c + 1))
                        # if rand_data <= (1 / alpha):
                        #     betaq = pow(rand_data * alpha, 1 / (eta_c + 1))
                        # else:
                        #     betaq = pow(1 / (2 - rand_data * alpha), 1 / (eta_c + 1))
                        if rand_data < 0.5:
                            betaq = pow(2 * rand_data, 1 / (eta_c + 1))
                        else:
                            betaq = pow(1 / (2 - 2 * rand_data), 1 / (eta_c + 1))
                        c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

                        # beta = 1 + (2 * (yu - y2) / (y2 - y1))
                        # alpha = 2 - pow(beta, - (eta_c + 1))
                        # if rand_data <= (1 / alpha):
                        #     betaq = pow(rand_data * alpha, 1 / (eta_c + 1))
                        # else:
                        #     betaq = pow(1 / (2 - rand_data * alpha), 1 / (eta_c + 1))
                        rand_data = get_random(0, 1)
                        if rand_data < 0.5:
                            betaq = pow(2 * rand_data, 1 / (eta_c + 1))
                        else:
                            betaq = pow(1 / (2 - 2 * rand_data), 1 / (eta_c + 1))
                        c2 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                        if c1 < yl:
                            c1 = yl
                        elif c1 > yu:
                            c1 = yu

                        if c2 < yl:
                            c2 = yl
                        elif c2 > yu:
                            c2 = yu

                        if get_random(0, 1) <= 0.5:
                            child1.vars[i] = c2
                            child2.vars[i] = c1
                        else:
                            child1.vars[i] = c1
                            child2.vars[i] = c2
        return child1, child2

    def scatter_population(self, fig_name='fig.png', vmin=None, vmax=None, xlabel=None, ylabel=None, colorbarlabel=None):
        num_data = len(self.individuals[0].get_scatter_data())
        plt.figure()
        if num_data == 2:
            for ind in self.individuals:
                [x, y] = ind.get_scatter_data()
                plt.scatter(x, y, c='k', s=2)
        elif num_data == 3:
            x = []
            y = []
            z = []
            for ind in self.individuals:
                [c1, c2, c3] = ind.get_scatter_data()
                x.append(c1)
                y.append(c2)
                z.append(c3)
            plt.scatter(x, y, c='b', s=z)
        elif num_data == 4:
            x = []
            y = []
            color = []
            size = []
            for ind in self.individuals:
                [c1, c2, c3, c4] = ind.get_scatter_data()
                x.append(c1)
                y.append(c2)
                color.append(c3)
                size.append(c4)
            plt.scatter(x, y, c=color, s=size, vmin=vmin, vmax=vmax)
            plt.colorbar(label=colorbarlabel)
        plt.title(fig_name)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(fig_name)
        plt.close()

    def output_population(self, filename):
        file1 = open(filename, 'w')
        file1.write(self.individuals[0].the_first_line())
        for i in range(self.rank_counter):
            file1.write(f'\n\n----------------------------------------------\nrank {i + 1}\n')
            for ind in self.individuals:
                if ind.rank == i + 1:
                    file1.write(f'{ind}\n')
        file1.close()
