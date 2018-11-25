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

    def update(self, representative, members):
        self.root = root
        self.members = members

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

        compatibility_threshold = self.species_set_config.compatibility_threshold

        # Find the best representatives for each existing species.
        unspeciated = set(iterkeys(population))
        distances = GenomeDistanceCache(config.genome_config)
        new_representatives = {}
        new_members = {}
        for sid, s in iteritems(self.species):
            candidates = []
            for gid in unspeciated:
                g = population[gid]
                d = distances(s.representative, g)
                candidates.append((d, g))

            # The new representative is the genome closest to the current representative.
            ignored_rdist, new_rep = min(candidates, key=lambda x: x[0])
            new_rid = new_rep.key
            new_representatives[sid] = new_rid
            new_members[sid] = [new_rid]
            unspeciated.remove(new_rid)

        # Partition population into species based on genetic similarity.
        while unspeciated:
            gid = unspeciated.pop()
            g = population[gid]

            # Find the species with the most similar representative.
            candidates = []
            for sid, rid in iteritems(new_representatives):
                rep = population[rid]
                d = distances(rep, g)
                if d < compatibility_threshold:
                    candidates.append((d, sid))

            if candidates:
                ignored_sdist, sid = min(candidates, key=lambda x: x[0])
                new_members[sid].append(gid)
            else:
                # No species is similar enough, create a new species, using
                # this genome as its representative.
                sid = next(self.indexer)
                new_representatives[sid] = gid
                new_members[sid] = [gid]

        # Update species collection based on new speciation.
        self.genome_to_species = {}
        for sid, rid in iteritems(new_representatives):
            s = self.species.get(sid)
            if s is None:
                s = Species(sid, generation)
                self.species[sid] = s

            members = new_members[sid]
            for gid in members:
                self.genome_to_species[gid] = sid

            member_dict = dict((gid, population[gid]) for gid in members)
            s.update(population[rid], member_dict)

        gdmean = mean(itervalues(distances.distances))
        gdstdev = stdev(itervalues(distances.distances))
        self.reporters.info(
            'Mean genetic distance {0:.3f}, standard deviation {1:.3f}'.format(gdmean, gdstdev))

    def get_species_id(self, individual_id):
        return self.genome_to_species[individual_id]

    def get_species(self, individual_id):
        sid = self.genome_to_species[individual_id]
        return self.species[sid]
