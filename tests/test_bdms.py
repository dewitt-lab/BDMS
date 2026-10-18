import dataclasses
import numpy as np
import ete4
import bdms
import unittest


class TestTree(unittest.TestCase):
    def setUp(self):
        self.tree = bdms.Tree()
        self.tree.state = 0.0
        for seed in range(1000):
            try:
                self.tree.evolve(
                    5,
                    bdms.poisson.ConstantProcess(1),
                    bdms.poisson.ConstantProcess(1),
                    bdms.poisson.ConstantProcess(1),
                    bdms.mutators.GaussianMutator(-1, 1),
                    min_survivors=20,
                    seed=seed,
                )
                break
            except bdms.tree.TreeError:
                continue

    def test_sample_survivors(self):
        self.tree.sample_survivors(n=10)
        self.assertTrue(
            all(leaf.event in ("survival", "sampling", "death") for leaf in self.tree)
        )
        self.assertTrue(
            all(
                len(node.children) == 2
                for node in self.tree.traverse()
                if node.event == "birth"
            )
        )
        mean_root_to_tip_distance = np.mean(
            np.array(
                [
                    self.tree.get_distance(self.tree, leaf)
                    for leaf in self.tree
                    if leaf.event == "sampling"
                ]
            )
        )
        for leaf in self.tree:
            if leaf.event == "sampling":
                self.assertAlmostEqual(
                    self.tree.get_distance(self.tree, leaf),
                    mean_root_to_tip_distance,
                    places=5,
                )

    def test_prune_unsampled(self):
        self.tree.sample_survivors(n=10)
        original_sampled = set(
            [node for node in self.tree.traverse() if node.event == "sampling"]
        )
        self.tree.prune_unsampled()
        self.assertTrue(all(leaf.event == "sampling" for leaf in self.tree))
        mutation_count = sum(node.event == "mutation" for node in self.tree.traverse())
        self.tree.remove_mutation_events()
        self.assertTrue(
            all(
                len(node.children) == 2
                for node in self.tree.traverse()
                if node.event == "birth"
            )
        )
        mutation_count2 = sum(node.n_mutations for node in self.tree.traverse())
        self.assertEqual(mutation_count, mutation_count2)
        self.assertTrue(
            not any(len(node.children) == 1 for node in self.tree.descendants())
        )
        self.assertEqual(
            set([node for node in self.tree.traverse() if node.event == "sampling"]),
            original_sampled,
        )
        mean_root_to_tip_distance = np.mean(
            np.array([self.tree.get_distance(self.tree, leaf) for leaf in self.tree])
        )
        for leaf in self.tree:
            self.assertAlmostEqual(
                self.tree.get_distance(self.tree, leaf),
                mean_root_to_tip_distance,
                places=5,
            )

    def test_node_props(self):
        # node data are ETE props, so they are written to newick
        newick = self.tree.write(
            props=["t", "event", "state"], parser=1, format_root_node=True
        )
        tree2 = ete4.Tree(newick, parser=1)
        for node, node2 in zip(self.tree.traverse(), tree2.traverse(), strict=True):
            self.assertEqual(node.name, node2.name)
            self.assertEqual(node.t, float(node2.props["t"]))
            self.assertEqual(str(node.event), node2.props["event"])
            self.assertEqual(node.state, float(node2.props["state"]))
        # newick can only be read into a plain ete4.Tree
        with self.assertRaises(TypeError):
            bdms.Tree(newick)

    def test_attribute_props(self):
        node = bdms.Tree(state=0)
        self.assertEqual(node.name, "0")
        self.assertEqual(node.dist, 0)
        self.assertEqual(node.props["state"], 0)
        node.x = 1.0
        self.assertEqual(node.props["x"], 1.0)
        del node.x
        self.assertNotIn("x", node.props)
        self.assertFalse(hasattr(node, "x"))
        node._x = 1.0
        self.assertNotIn("_x", node.props)
        self.assertEqual(node._x, 1.0)

    def test_copy(self):
        self.tree.sample_survivors(n=10)
        tree2 = self.tree.copy()
        for node, node2 in zip(self.tree.traverse(), tree2.traverse(), strict=True):
            self.assertIsInstance(node2, bdms.Tree)
            self.assertEqual(node.props, node2.props)
            self.assertTrue(node2._sampled)
        # pruning the copy doesn't affect the original
        n_leaves = len(self.tree)
        tree2.prune_unsampled()
        self.assertEqual(len(tree2), 10)
        self.assertEqual(len(self.tree), n_leaves)
        self.assertFalse(self.tree._pruned)


class TestStates(unittest.TestCase):
    def test_tuple_states(self):
        # the default death and mutation processes have rate zero
        tree = bdms.Tree(state=("a", 0))
        tree.evolve(
            2.0, birth_process=bdms.poisson.DiscreteProcess({("a", 0): 1.0}), seed=0
        )
        scalar_tree = bdms.Tree(state="a0")
        scalar_tree.evolve(
            2.0, birth_process=bdms.poisson.DiscreteProcess({"a0": 1.0}), seed=0
        )
        self.assertEqual(
            [node.t for node in tree.traverse()],
            [node.t for node in scalar_tree.traverse()],
        )
        states = [("a", 0), ("a", 1), ("b", 0)]
        rates = bdms.poisson.ConstantProcess(1.0).λ_homogeneous(states)
        self.assertEqual(rates.shape, (3,))

    def test_unordered_states(self):
        # states that can't be compared, two of which can't give birth
        @dataclasses.dataclass(frozen=True)
        class State:
            kind: str

        a, b, c = State("a"), State("b"), State("c")
        tree = bdms.Tree(state=a)
        tree.evolve(
            6.0,
            birth_process=bdms.poisson.DiscreteProcess({a: 0.0, b: 0.0, c: 2.0}),
            mutation_process=bdms.poisson.ConstantProcess(1.0),
            mutator=bdms.mutators.DiscreteMutator(
                (a, b, c), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
            ),
            init_population=3,
            seed=0,
        )
        self.assertTrue(any(node.event == "birth" for node in tree.traverse()))


class TestCapacity(unittest.TestCase):
    capacity = 100
    t = 20.0

    def evolve(self, capacity_method):
        """Return the survivors of a growing population with this capacity method."""
        tree = bdms.Tree(state=0)
        tree.evolve(
            self.t,
            birth_process=bdms.poisson.ConstantProcess(2.0),
            death_process=bdms.poisson.ConstantProcess(1.0),
            capacity=self.capacity,
            capacity_method=capacity_method,
            init_population=10,
            seed=0,
        )
        return [leaf for leaf in tree if leaf.event == "survival"]

    def test_exceeded(self):
        tree = bdms.Tree(state=0)
        with self.assertRaises(bdms.TreeError):
            tree.evolve(self.t, capacity=self.capacity, seed=0)
        # the aborted simulation is removed
        self.assertEqual(len(tree.children), 0)

    def test_stop(self):
        survivors = self.evolve("stop")
        self.assertEqual(len(survivors), self.capacity)
        end_times = {leaf.t for leaf in survivors}
        self.assertEqual(len(end_times), 1)
        self.assertLess(end_times.pop(), self.t)

    def test_hard(self):
        n_survivors = len(self.evolve("hard"))
        self.assertLessEqual(n_survivors, self.capacity)
        self.assertGreater(n_survivors, self.capacity / 2)

    def test_birth_and_death(self):
        for capacity_method in ("birth", "death"):
            with self.subTest(capacity_method=capacity_method):
                n_survivors = len(self.evolve(capacity_method))
                self.assertGreater(n_survivors, self.capacity / 2)
                self.assertLess(n_survivors, 2 * self.capacity)

    def test_zero_total_rate(self):
        # cells divide only in the last of K cell-cycle stages, so the total birth
        # rate is zero until a cell reaches that stage
        K = 3
        birth = bdms.poisson.DiscreteProcess(
            {k: (K if k == K - 1 else 0.0) for k in range(K)}
        )
        progression = bdms.poisson.DiscreteProcess(
            {k: (K if k < K - 1 else 0.0) for k in range(K)}
        )
        transitions = np.zeros((K, K))
        for k in range(K - 1):
            transitions[k, k + 1] = 1.0
        transitions[K - 1, 0] = 1.0
        death_processes = (
            bdms.poisson.ConstantProcess(0.5),
            bdms.poisson.DiscreteProcess({k: 0.5 for k in range(K)}),
        )
        for death in death_processes:
            with self.subTest(death_process=death):
                tree = bdms.Tree(state=0)
                tree.evolve(
                    10.0,
                    birth_process=birth,
                    death_process=death,
                    mutation_process=progression,
                    mutator=bdms.mutators.DiscreteMutator(tuple(range(K)), transitions),
                    birth_mutation_prob=1.0,
                    capacity=self.capacity,
                    capacity_method="birth",
                    seed=0,
                )
                self.assertTrue(any(node.event == "birth" for node in tree.traverse()))

    def test_zero_rate_multiplier(self):
        process = bdms.poisson.ConstantProcess(1.0)
        waiting_time = process.waiting_time_rv(0, 0.0, rate_multiplier=0.0, seed=0)
        self.assertEqual(waiting_time, np.inf)


if __name__ == "__main__":
    unittest.main()
