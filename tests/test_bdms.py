import numpy as np
import bdms
import unittest


class TestTreeNode(unittest.TestCase):
    def setUp(self):
        self.tree = bdms.TreeNode()
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
        # node data are ETE props, so they are written to (and read from) newick
        newick = self.tree.write(
            props=["t", "event", "state"], parser=1, format_root_node=True
        )
        tree2 = bdms.TreeNode(newick, parser=1)
        for node, node2 in zip(self.tree.traverse(), tree2.traverse(), strict=True):
            self.assertIsInstance(node2, bdms.TreeNode)
            self.assertEqual(node.name, node2.name)
            self.assertEqual(node.t, float(node2.t))
            self.assertEqual(str(node.event), node2.event)
            self.assertEqual(node.state, float(node2.state))
            self.assertEqual(node2.n_mutations, 0)

    def test_attribute_props(self):
        node = bdms.TreeNode(state=0)
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
            self.assertIsInstance(node2, bdms.TreeNode)
            self.assertEqual(node.props, node2.props)
            self.assertTrue(node2._sampled)
        # pruning the copy doesn't affect the original
        n_leaves = len(self.tree)
        tree2.prune_unsampled()
        self.assertEqual(len(tree2), 10)
        self.assertEqual(len(self.tree), n_leaves)
        self.assertFalse(self.tree._pruned)

    def test_deep_tree(self):
        # deeper than the C stack allows for recursive tree methods
        tree = bdms.TreeNode()
        node = tree
        for i in range(50_000):
            node = node.add_child(bdms.TreeNode(name=i + 1))
            node.event = "mutation"
        node.event = "sampling"
        tree._sampled = True
        tree.prune_unsampled()
        tree.remove_mutation_events()
        self.assertEqual(len(tree.children), 1)
        self.assertEqual(tree.children[0].n_mutations, 50_000 - 1)


if __name__ == "__main__":
    unittest.main()
