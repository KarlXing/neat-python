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
class PedigreeSpecies(object):
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
        #self.alive_members = {}
        self.all_members = {}

    def update(self, root, all_members):
        self.root = root
        self.all_members = all_members
        self.root_family_generation = all_members[root].family_generation
        self.update_alive_members()

    def update_alive_members(self):
        self.members = {}
        for gid, g in iteritems(self.all_members):
            if g.alive:
                self.members[gid] = g

    def get_fitnesses(self):
        return [m.fitness for m in itervalues(self.members)]

    def clean(self, population):
        assert(isinstance(population, dict))
        dead_genomes = set(iterkeys(population))
        # print("in clean ", dead_genomes)
        for gid in dead_genomes:
            # print("gid ,", gid)
            assert(gid in self.members)
            g = self.all_members[gid]
            # change state to dead
            g.killed()
            # remove itself from its parent's children list
            self.sweep(gid)
            # members = []
            # for k in iterkeys(self.members):
            #     members.append(k)
            # print("after clean gid ", members)

    def sweep(self, gid):
        g = self.all_members[gid]
        if len(g.children) > 0 or g.alive or gid == self.root:
            return
        pid = g.parent
        del self.all_members[gid]
        p = self.all_members[pid]
        p.children.remove(gid)
        self.sweep(pid)

    def get_offspring(self, gid):
        offspring = {}
        g = self.all_members[gid]
        for child in g.children:
            offspring[child] = self.all_members[child]
            offspring.update(self.get_offspring(child))
        return offspring

    def get_new_roots(self, root):
        new_roots = set()
        for gid in self.all_members[root].children:
            g = self.all_members[gid]
            if g.alive:
                new_roots.add(gid)
            else:
                new_roots.update(self.get_new_roots(gid))
        return new_roots

    def show(self):
        print("all members")
        for gid, g in iteritems(self.all_members):
            print(gid, " children: ", g.children )
        print("alive members")
        for gid, g in iteritems(self.members):
            print(gid, " children: ", g.children )

class PedigreeSpeciesSet(DefaultClassConfig):
    """ Encapsulates the default speciation scheme. """

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.species_set_config = config
        self.reporters = reporters
        self.indexer = count(1)
        self.species = {}
        self.genome_to_species = {}

    def init_with_roots(self, roots, generation):
        self.species = {}
        for gid, g in iteritems(roots):
            g.family_generation = 0
            sid = next(self.indexer)
            s = PedigreeSpecies(sid, generation)
            all_members = {}
            all_members[gid] = g
            s.update(gid, all_members)
            self.species[sid] = s
            self.genome_to_species[gid] = sid

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('compatibility_threshold', float),
                                   ConfigParameter('initial_species', int, 10),
                                   ConfigParameter('max_family_generation', int, 10)])

    def speciate(self, config, population, generation):
        # print("genome to species: ", self.genome_to_species)
        """
        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        # print("population")
        # for k,v in population.items():
        #     print(k, v.parent)

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
                assert(s.all_members[gid].alive)
                new_members[sid].append(gid)
            else:
                allocated = False
                for sid, s in iteritems(self.species):
                    root = s.all_members[s.root]
                    # print(sid, " members are : ", s.members)
                    # print(g.parent, g.parent in s.members)
                    if g.parent in s.all_members:
                        allocated = True
                        new_members[sid].append(gid)
                        parent = s.all_members[g.parent]
                        parent.add_child(gid)
                        s.all_members[gid] = g
                        if (parent.family_generation - root.family_generation + 1) > \
                            config.species_set_config.max_family_generation:
                            s.need_split = True
                        break
                if not allocated:
                    raise RuntimeError("genome was not allocated to any species")

        # mark all genomes absent from new_members as dead and clean them
        for sid, s in iteritems(self.species):
            alive_members = new_members[sid]
            dead_genomes = {}
            for gid, g in iteritems(s.members):
                if gid not in alive_members:
                    dead_genomes[gid] = g
            dead = []
            for k in iterkeys(dead_genomes):
                dead.append(k)
            print(sid, "dead genomes ", dead)
            # print("before clean")
            # s.show()
            s.clean(dead_genomes)
            s.update_alive_members()

        # update species if reach family generation limit
        # during species spliting, the original root was dropped no matter whether it's alive
        new_species = {}
        for sid, s in iteritems(self.species):
            if s.need_split is True:
                unspeciated = iterkeys(s.all_members)
                new_roots = s.get_new_roots(s.root)
                for root in new_roots:
                    new_sid = next(self.indexer)
                    new_s = PedigreeSpecies(new_sid, generation)
                    new_members = {}
                    new_members[root] = s.members[root]
                    new_members.update(s.get_offspring(root))
                    new_s.update(root, new_members)
                    new_species[new_sid] = new_s
            else:
                new_species[sid] = s
        self.species = new_species


        # update species collection
        self.genome_to_species = {}
        for sid, s in iteritems(self.species):
            for gid in iterkeys(s.members):
                self.genome_to_species[gid] = sid

    def get_species_id(self, individual_id):
        return self.genome_to_species[individual_id]

    def get_species(self, individual_id):
        sid = self.genome_to_species[individual_id]
        return self.species[sid]
