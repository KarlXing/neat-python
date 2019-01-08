# Evolve a control/reward estimation network for the OpenAI Gym
# LunarLander-v2 environment (https://gym.openai.com/envs/LunarLander-v2).
# Sample run here: https://gym.openai.com/evaluations/eval_FbKq5MxAS9GlvB7W6ioJkg

from __future__ import print_function

import gym
import gym.wrappers

import matplotlib.pyplot as plt

import multiprocessing
import neat
import numpy as np
import os
import pickle
import random
import time
import datetime
from tensorboardX import SummaryWriter
# import visualize
import sys
sys.path.append("../../../")

from neatpython.neat.genome_pedigree import PedigreeGenome
from neatpython.neat.population_pedigree import PedigreePopulation
from neatpython.neat.reproduction_ep import EPReproduction
from neatpython.neat.species_pedigree import PedigreeSpeciesSet

NUM_CORES = 8

env = gym.make('MsPacman-ram-v0')

print("action space: {0!r}".format(env.action_space))
print("observation space: {0!r}".format(env.observation_space))

#env = gym.wrappers.Monitor(env, 'results', force=True)


def compute_fitness(genome, net, episodes, min_reward, max_reward):
    m = int(round(np.log(0.01) / np.log(genome.discount)))
    discount_function = [genome.discount ** (m - i) for i in range(m + 1)]

    reward_error = []
    for score, data in episodes:
        # Compute normalized discounted reward.
        dr = np.convolve(data[:,-1], discount_function)[m:]
        dr = 2 * (dr - min_reward) / (max_reward - min_reward) - 1.0
        dr = np.clip(dr, -1.0, 1.0)

        for row, dr in zip(data, dr):
            observation = row[:8]
            action = int(row[8])
            output = net.activate(observation)
            reward_error.append(float((output[action] - dr) ** 2))

    return reward_error


class PooledErrorCompute(object):
    def __init__(self):
        self.pool = None if NUM_CORES < 2 else multiprocessing.Pool(NUM_CORES)
        self.test_episodes = []
        self.generation = 0

        self.min_reward = -200
        self.max_reward = 200

        self.episode_score = []
        self.episode_length = []

    def simulate(self, nets):
        scores = []
        for genome, net in nets:
            data = []
            for i in range(5):
                observation = env.reset()
                step = 0
                while 1:
                    step += 1
                    if step < 200 and random.random() < 0.2:
                        action = env.action_space.sample()
                    else:
                        output = net.activate(observation)
                        action = np.argmax(output)
                # output = net.activate(observation)
                # action = np.argmax(output)

                    observation, reward, done, info = env.step(action)
                    data.append(np.hstack((observation, action, reward)))

                    if done:
                        break

            data = np.array(data)
            score = np.sum(data[:,-1])/5
            self.episode_score.append(score)
            scores.append(score)
            genome.fitness = score+600
            self.episode_length.append(step)


    def evaluate_genomes(self, genomes, config):
        self.generation += 1

        t0 = time.time()
        nets = []
        for gid, g in genomes:
            nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))

        #print("network creation time {0}".format(time.time() - t0))
        t0 = time.time()
        self.simulate(nets)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    writer = SummaryWriter()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'pedigreeconfig')
    config = neat.Config(PedigreeGenome, EPReproduction,
                         PedigreeSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = PedigreePopulation(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 25 generations or 900 seconds.
    # pop.add_reporter(neat.Checkpointer(100, 9000))

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    ec = PooledErrorCompute()
    #pop.fitness_calculate(ec.evaluate_genomes)
    pop.fitness_calculate(ec.evaluate_genomes)
    id = str(datetime.datetime.now())
    figfile = "Pedigree_"+id+".svg"
    g_step = 0
    while 1:
        g_step += 1
        try:
            t0 = time.time()
            gen_best = pop.run(ec.evaluate_genomes, 1)

            #visualize.plot_stats(stats, ylog=False, view=False, filename=figfile)
            # starts to check whether the question is solved after the 5th generation
            all_fit = pop.get_all_fitness()
            writer.add_scalar('analysis/mean fitness', sum(all_fit)/len(all_fit), g_step)
            writer.add_scalar('analysis/best fitness', stats.most_fit_genomes[-1].fitness, g_step)

            complexity = pop.get_complexity()
            writer.add_scalar('analysis/nodes', complexity[0], g_step)
            writer.add_scalar('analysis/connections', complexity[1], g_step)
            writer.add_scalar('analysis/species', len(pop.species.species), g_step)
            if g_step < 5:
                continue

            if g_step % 5 == 0:
                pop.fitness_calculate(ec.evaluate_genomes)
            #print("Average mean fitness over last 5 generations: {0}".format(mfs))

            #mfs = sum(stats.get_fitness_stat(min)[-5:]) / 5.0
            #print("Average min fitness over last 5 generations: {0}".format(mfs))

            # Use the best genomes seen so far as an ensemble-ish control system.
            # best_genomes = stats.best_unique_genomes(5)
            # best_networks = []
            # for g in best_genomes:
            #     best_networks.append(neat.nn.FeedForwardNetwork.create(g, config))
            # solved = True
            # best_scores = []
            # for k in range(100):
            #     observation = env.reset()
            #     score = 0
            #     step = 0
            #     while 1:
            #         step += 1
            #         # Use the total reward estimates from all five networks to
            #         # determine the best action given the current state.
            #         votes = np.zeros((9,))
            #         for n in best_networks:
            #             output = n.activate(observation)
            #             votes[np.argmax(output)] += 1

            #         best_action = np.argmax(votes)
            #         observation, reward, done, info = env.step(best_action)
            #         score += reward
            #         #env.render()
            #         if done:
            #             break

            #     ec.episode_score.append(score)
            #     ec.episode_length.append(step)
            #     best_scores.append(score)
            #     avg_score = sum(best_scores) / len(best_scores)
            #     if avg_score < 200:
            #         writer.add_scalar("analysis/passed", k, g_step)
            #         solved = False
            #         break

            # if solved:
            #     print("Solved.")

            #     # Save the winners.
            #     # for n, g in enumerate(best_genomes):
            #     #     name = 'winner-{0}'.format(n)
            #     #     with open(name+'.pickle', 'wb') as f:
            #     #         pickle.dump(g, f)

            #         # visualize.draw_net(config, g, view=False, filename=name+"-net.gv")
            #         # visualize.draw_net(config, g, view=False, filename=name+"-net-enabled.gv",
            #         #                    show_disabled=False)
            #         # visualize.draw_net(config, g, view=False, filename=name+"-net-enabled-pruned.gv",
            #         #                    show_disabled=False, prune_unused=True)

            #     break
        except KeyboardInterrupt:
            print("User break.")
            break
    
    stats_file = "PedigreeStats_"+id+".p"
    pickle.dump(stats, open(stats_file, "wb"))
    writer.close()
    env.close()


if __name__ == '__main__':
    run()