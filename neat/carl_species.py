"""Divides the population into species based on genomic distances."""
from itertools import count

from neat.math_util import mean, stdev
from neat.six_util import iteritems, iterkeys, itervalues
from neat.config import ConfigParameter, DefaultClassConfig

"""
Ancestry information is stored in genomes. root is the root genome id.
members dict still contain not only alive genomes, but also some dead genomes.
max_family_generation is the maximum family generation of genomes which will 
cause species spilting once it exceeds the limit.
"""
class NewSpecies(object):
    def __init__(self, key, generation):
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.root = None
        self.members = {}
        self.fitness = None
        self.adjusted_fitness = None
        self.fitness_history = []
        self.need_split = False
        self.alive_members = {}

    def update(self, root, members):
        self.root = root
        self.members = members
        self.root_family_generation = members[root].family_generation

    def update_alive_members(self):
        self.alive_members = {}
        for gid, g in iteritems(self.members):
            if g.alive:
                self.alive_members[gid] = g

    def get_fitnesses(self):
        return [m.fitness for m in itervalues(self.members)]

    def clean(self, population):
        assert(isinstance(population, dict))
        dead_genomes = set(iterkeys(population))
        for gid in dead_genomes:
            assert(gid in self.members)
            g = self.members[gid]
            # change state to dead
            g.killed()
            # remove itself from its parent's children list
            if gid != root:
                parent = self.members[g.parent]
                parent.children.pop(gid)
                sweep(gid)

    def sweep(self, gid):
        g = self.members[gid]
        if len(g.children) > 0 or g.alive or gid == self.root:
            return
        pid = g.parent
        del self.members[gid]
        sweep(self, pid)

    def get_offspring(self, gid):
        offspring = {}
        g = self.members[gid]
        for child in g.children:
            offspring[child] = self.members[child]
            offspring.update(get_offspring(self, child))
        return offspring

class DefaultSpeciesSet(DefaultClassConfig):
    """ Encapsulates the default speciation scheme. """

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.species_set_config = config
        self.reporters = reporters
        self.indexer = count(1)
        self.species = {}
        self.genome_to_species = {}

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('compatibility_threshold', float)])

    def speciate(self, config, population, generation):
        """
        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        assert isinstance(population, dict)
        # allocate population to species by parent or itself
        unspeciated = set(iterkeys(population))
        new_members = {} # keep track of alive genomes for each species
        for sid in iterkeys(self.species):
            new_members[sid] = []

        while unspeciated:
            gid = unspeciated.pop()
            g = population[gid]
            if gid in self.genome_to_species:
                sid = self.genome_to_species[gid]
                s = self.species[sid]
                assert(s.members[gid].alive)
                new_members[sid].append(gid)
            else:
                for sid, s in iteritems(self.species):
                    if g.parent in s.members:
                        new_members[sid].append(gid)
                        parent = s.members[g.parent]
                        parent.add_child(gid)
                        s.members[gid] = g
                        if (parent.family_generation - self.root_family_generation + 1) >
                                config.species_set_config.family_generation:
                            s.need_split = True
                        break
                raise RuntimeError("genome was not allocated to any species")

        # mark all genomes absent from new_members as dead and clean them
        for sid, s in iteritems(self.species):
            members = new_members[sid]
            dead_genomes = {}
            for gid, g in iteritems(s.members):
                if gid not in members:
                    g.killed()
                    dead_genomes[gid] = g
            s.clean(dead_genomes)

        # update species if reach family generation limit
        # during species spliting, the original root was dropped no matter whether it's alive
        for sid, s in iteritems(self.species):
            if s.need_split = True:
                unspeciated = iterkeys(s.members)
                new_roots = s.members[s.root].children
                for root in new_roots:
                    new_sid = next(self.indexer)
                    new_s = NewSpecies(new_sid, generation)
                    new_members = {}
                    new_members[root] = s.members[root]
                    new_members.update(s.get_offspring(root))
                    new_s.update(root, new_members)
                    self.species[new_sid] = new_s
                del self.species[sid]


        # update species collection
        self.genome_to_species = {}
        for sid, s in iteritems(self.species):
            for gid in iterkeys(s.members):
                self.genome_to_species[gid] = sid


        # #compatibility_threshold = self.species_set_config.compatibility_threshold

        # # Find the best representatives for each existing species.
        # unspeciated = set(iterkeys(population))
        # distances = GenomeDistanceCache(config.genome_config)
        # new_representatives = {}
        # new_members = {}
        # for sid, s in iteritems(self.species):
        #     candidates = []
        #     for gid in unspeciated:
        #         g = population[gid]
        #         d = distances(s.representative, g)
        #         candidates.append((d, g))

        #     # The new representative is the genome closest to the current representative.
        #     ignored_rdist, new_rep = min(candidates, key=lambda x: x[0])
        #     new_rid = new_rep.key
        #     new_representatives[sid] = new_rid
        #     new_members[sid] = [new_rid]
        #     unspeciated.remove(new_rid)

        # # Partition population into species based on genetic similarity.
        # while unspeciated:
        #     gid = unspeciated.pop()
        #     g = population[gid]

        #     # Find the species with the most similar representative.
        #     candidates = []
        #     for sid, rid in iteritems(new_representatives):
        #         rep = population[rid]
        #         d = distances(rep, g)
        #         if d < compatibility_threshold:
        #             candidates.append((d, sid))

        #     if candidates:
        #         ignored_sdist, sid = min(candidates, key=lambda x: x[0])
        #         new_members[sid].append(gid)
        #     else:
        #         # No species is similar enough, create a new species, using
        #         # this genome as its representative.
        #         sid = next(self.indexer)
        #         new_representatives[sid] = gid
        #         new_members[sid] = [gid]

        # # Update species collection based on new speciation.
        # self.genome_to_species = {}
        # for sid, rid in iteritems(new_representatives):
        #     s = self.species.get(sid)
        #     if s is None:
        #         s = Species(sid, generation)
        #         self.species[sid] = s

        #     members = new_members[sid]
        #     for gid in members:
        #         self.genome_to_species[gid] = sid

        #     member_dict = dict((gid, population[gid]) for gid in members)
        #     s.update(population[rid], member_dict)

        # gdmean = mean(itervalues(distances.distances))
        # gdstdev = stdev(itervalues(distances.distances))
        # self.reporters.info(
        #     'Mean genetic distance {0:.3f}, standard deviation {1:.3f}'.format(gdmean, gdstdev))

    def get_species_id(self, individual_id):
        return self.genome_to_species[individual_id]

    def get_species(self, individual_id):
        sid = self.genome_to_species[individual_id]
        return self.species[sid]
