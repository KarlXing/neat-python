"""Carl Genome Implementation"""
"""Add children, parent, family_generation ,alive attributes"""
import neat

from neat.genome import DefaultGenome
import random


class CarlLanderGenome(DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.discount = None
        self.children = []
        self.parent = None
        self.family_generation = None
        self.alive = True

    def configure_new(self, config):
        super().configure_new(config)
        self.discount = 0.01 + 0.98 * random.random()

    def configure_crossover(self, genome1, genome2, config):
        super().configure_crossover(genome1, genome2, config)
        self.discount = random.choice((genome1.discount, genome2.discount))

    def mutate(self, config):
        super().mutate(config)
        self.discount += random.gauss(0.0, 0.05)
        self.discount = max(0.01, min(0.99, self.discount))

    def distance(self, other, config):
        dist = super().distance(other, config)
        disc_diff = abs(self.discount - other.discount)
        return dist + disc_diff

    def set_parent(self, parent1, parent1_family_generation, parent2=None, parent2_family_generation = None):
        if parent2 is None:
            self.parent = parent1
            self.family_generation = parent1_family_generation + 1
        else:
            self.parent = random.choice((parent1, parent2))
            if self.parent == parent1:
                self.family_generation = parent1_family_generation + 1
            else:
                self.family_generation = parent2_family_generation + 1

    def show(self):
        print("discount: ", self.discount)
        print("children: ", self.children)
        print("parent: ", self.parent)
        print("family_generation: ", self.family_generation)
        print("alive: ", self.alive)

    def add_child(self, child):
        self.children.append(child)

    def set_family_generation(self, generation):
        self.family_generation = generation

    def killed(self):
        self.alive = False

    def get_alive(self):
        return self.alive

    def get_num_children(self):
        return len(self.children)

    def get_parent(self):
        return self.parent

    def __str__(self):
        return "Reward discount: {0}\n{1}".format(self.discount,
                                                  super().__str__())
