"""Tests for layout2graph.py

Author: antfdez@uvigo.es
"""

import networkx as nx
import unittest

from layout2graph import layout2graph


class TestStrToNum(unittest.TestCase):
    
    def test_str2num_integer(self):
        strings = ['-1000', '-5', '0', '1', '99']
        expected = [-1000, -5, 0, 1, 99]
        got = [layout2graph.str2num(s) for s in strings]
        self.assertEqual(got, expected)

    def test_str2num_integer_as_float(self):
        strings = ['-10.0', '0.0', '5.0']
        expected = [-10, 0, 5]
        got = [layout2graph.str2num(s) for s in strings]
        self.assertEqual(got, expected)

    def test_str2num_float(self):
        strings = ['-100.9', '-.5' ,'-0.4', '.2', '0.3', '11.7']
        expected = [-100.9, -0.5, -0.4, 0.2, 0.3, 11.7]
        got = [layout2graph.str2num(s) for s in strings]
        self.assertEqual(got, expected)

    def test_str2num_text(self):
        self.assertRaises(ValueError, layout2graph.str2num, 'one')


class PruneTestCase(unittest.TestCase):
    """Base class for all Prune tests."""
    
    def assertPrunedEqual(self, nod1, edg1, nod2, edg2, essential, required):

        complete = nx.Graph()
        complete.add_nodes_from(nod1)
        complete.add_edges_from(edg1)
        got = layout2graph.prune(complete, essential, required)

        expected = nx.Graph()
        expected.add_nodes_from(nod2)
        expected.add_edges_from(edg2)
        
        self.assertEqual(set(got.nodes), set(expected.nodes))
        self.assertEqual(set(got.edges), set(expected.edges))
        


class TestPrune(PruneTestCase):
    
    def setUp(self):
        self.essential = {1, 2, 3, 4, 5, 6, 7, 8}
        self.required = {9, 10}
    
    
    def test_prune_campervan_01(self):
        nodes_complete = [
            1, 2, 4, 5, 6, 7,
            9.1, 9.2, 9.3, 9.4, 10.1, 10.2, 11, 14, 15, 16, 20,
            ]
        edges_complete = [
            (1, 2), (1, 4), (1, 5), (1, 9.3), (1, 9.4), (1, 10.1), 
            (1, 10.2), (1, 11),
            (4, 20),
            (6, 14),
            (7, 20), 
            (9.1, 20), (9.2, 20),
            (14, 15), (14, 16), (14, 20),
            ]
        nodes_pruned = [1, 2, 4, 5, 6, 7, 9, 10, 11, 14, 15, 16]
        edges_pruned = [(1, 2), (1, 4), (1, 5), (4, 7)]

        self.assertPrunedEqual(
            nodes_complete, edges_complete,nodes_pruned, edges_pruned,
            self.essential, self.required,
            )


    def test_prune_campervan_02(self):
        nodes_complete = [
            1, 2.1, 2.2, 2.3, 4, 5, 6, 7,
            8, 9.1, 9.2, 9.3, 9.4, 10, 11, 16, 18, 20,
            ]
        edges_complete = [
            (1, 2.1), (1, 2.2), (1, 4), (1, 5), (1, 8), (1, 10), (1, 16),
            (2.1, 2.2), (2.2, 4), (2.2, 18), (2.3, 4), (2.3, 8),
            (4, 18),
            (5, 11),
            (6, 7), (6, 8), (6, 9.4), (6, 20),
            (9.1, 10), (9.2, 10), (9.3, 9.4),
            ]
        nodes_pruned = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 16, 18]
        edges_pruned = [
            (1, 2), (1, 4), (1, 5), (1, 8),
            (2, 4), (2, 8),
            (6, 7), (6, 8),
            ]

        self.assertPrunedEqual(
            nodes_complete, edges_complete,nodes_pruned, edges_pruned,
            self.essential, self.required,
            )


    def test_prune_campervan_03(self):
        nodes_complete = [
            1, 2.1, 2.2, 4, 5, 6, 7,
            8, 9.1, 9.2, 10, 13, 15, 16, 19, 20,
            ]
        edges_complete = [
            (1, 2.1), (1, 2.2), (1, 4), (1, 16), (1, 19), (1, 20),
            (2.1, 4),
            (5, 19),
            (6, 20),
            (7, 20),
            (8, 19),
            (9.1, 20), (9.2, 20),
            (10, 20),
            (13, 20),
            (15, 20),
            ]
        nodes_pruned = [1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 19]
        edges_pruned = [(1, 2), (1, 4), (1, 6), (1, 7), (2, 4), (6, 7)]

        self.assertPrunedEqual(
            nodes_complete, edges_complete,nodes_pruned, edges_pruned,
            self.essential, self.required,
            )


    def test_prune_campervan_04(self):
        nodes_complete = [
            1, 2, 3, 4, 5, 6, 7,
            8.1, 8.2, 9.1, 9.2, 9.3, 10.1, 10.2, 11, 16, 20.1, 20.2,
            ]
        edges_complete = [
            (1, 2), (1, 3), (1, 4), (1, 5),
            (3, 5), (3, 6), (3, 7), (3, 16), (3, 20.1), (3, 20.2),
            (4, 8.1), (4, 8.2),
            (6, 7),
            (9.1, 20.2), (9.2, 20.2), (9.3, 20.2),
            (10.1, 20.1), (10.2, 20.1),
            (11, 20.1),
            ]
        nodes_pruned = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16]
        edges_pruned = [
            (1, 2), (1, 3), (1, 4), (1, 5),
            (3, 5), (3, 6), (3, 7),
            (4, 8),
            (6, 7),
            ]

        self.assertPrunedEqual(
            nodes_complete, edges_complete,nodes_pruned, edges_pruned,
            self.essential, self.required,
            )


    def test_prune_campervan_05(self):
        nodes_complete = [
            1, 2, 4, 5, 6, 7,
            8, 9.1, 9.2, 10.1, 10.2, 11, 14, 15, 16, 20.1, 20.2, 21,
            ]
        edges_complete = [
            (1, 2), (1, 4), (1, 5), (1, 8),
            (2, 20.1),
            (4, 10.1), (4, 11), (4, 20.1),
            (6, 20.1),
            (7, 20.1),
            (9.1, 20.2), (9.2, 20.2),
            (10.2, 20.1),
            (14, 20.1),
            (15, 20.1),
            (16, 20.1),
            (20.1, 20.2), (20.1, 21),
            ]
        nodes_pruned = [
            1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 21]
        edges_pruned = [
            (1, 2), (1, 4), (1, 5), (1, 8),
            (2, 4), (2, 6), (2, 7),
            (4, 6), (4, 7),
            (6, 7),
            ]

        self.assertPrunedEqual(
            nodes_complete, edges_complete,nodes_pruned, edges_pruned,
            self.essential, self.required,
            )


    def test_prune_campervan_06(self):
        nodes_complete = [1, 2, 4, 5, 6, 7, 8, 9.1, 9.2, 10.1, 10.2, 16]
        edges_complete = [
            (1, 4), (1, 5), (1, 7), (1, 10.1), (1, 10.2),
            (2, 4), (2, 8),
            (5, 6), (5, 16),
            (6, 7),
            (9.1, 10.1), (9.2, 10.2),
            ]
        nodes_pruned = [1, 2, 4, 5, 6, 7, 8, 9, 10, 16]
        edges_pruned = [
            (1, 4), (1, 5), (1, 7),
            (2, 4), (2, 8),
            (5, 6),
            (6, 7),
            ]

        self.assertPrunedEqual(
            nodes_complete, edges_complete,nodes_pruned, edges_pruned,
            self.essential, self.required,
            )


    def test_prune_campervan_07(self):
        nodes_complete = [
            1, 4, 5, 6, 7,
            8, 9.1, 9.2, 9.3, 9.4, 10.1, 10.2, 11, 12, 15, 16, 20]
        edges_complete = [
            (1, 4), (1, 5), (1, 10.1), (1, 10.2), (1, 15), (1, 16), (1, 20),
            (5, 8), (5, 12),
            (6, 7), (6, 20),
            (7, 11),
            (9.1, 20), (9.2, 20), (9.3, 10.1), (9.4, 10.2),
            (11, 20),
            ]
        nodes_pruned = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16]
        edges_pruned = [(1, 4), (1, 5), (1, 6), (5, 8), (6, 7)]
        self.assertPrunedEqual(
            nodes_complete, edges_complete,nodes_pruned, edges_pruned,
            self.essential, self.required,
            )


    def test_prune_campervan_10(self):
        nodes_complete = [
            1, 2.1, 2.2, 3, 4, 6, 7,
            9.1, 9.2, 9.3, 10.1, 10.2, 16, 20,
            ]
        edges_complete = [
            (1, 2.1), (1, 2.2), (1, 3), (1, 4), (1, 5), (1, 10.1), (1, 10.2),
            (2.2, 4),
            (3, 4), (3, 7), (3, 16),
            (6, 7),
            (9.1, 10.1), (9.2, 10.2), (9.3, 20),
            (16, 20),
            ]
        nodes_pruned = [1, 2, 3, 4, 6, 7, 9, 10, 16]
        edges_pruned = [
            (1, 2), (1, 3), (1, 4), (1, 5),
            (2, 4),
            (3, 4), (3, 7),
            (6, 7),
            ]

        self.assertPrunedEqual(
            nodes_complete, edges_complete,nodes_pruned, edges_pruned,
            self.essential, self.required,
            )


    def test_prune_campervan_11(self):
        nodes_complete = [
            1, 2.1, 2.2, 4, 6.1, 6.2, 7,
            9.1, 9.2, 9.3, 9.4, 10.1, 10.2, 11, 15, 16, 20.1, 20.2]
        edges_complete = [
            (1, 2.1), (1, 4), (1, 11), (1, 15), (1, 16), (1, 20.1),
            (2.1, 2.2), (2.1, 6.1),
            (6.1, 6.2),
            (7, 15),
            (9.1, 9.2), (9.2, 20.1), (9.3, 9.4), (9.3, 20.1),
            (10.1, 20.2), (10.2, 20.2),
            (20.1, 20.2),
            ]
        nodes_pruned = [1, 2, 4, 6, 7, 9, 10, 11, 15, 16]
        edges_pruned = [(1, 2), (1, 4), (2, 6)]

        self.assertPrunedEqual(
            nodes_complete, edges_complete,nodes_pruned, edges_pruned,
            self.essential, self.required,
            )


if __name__ == '__main__':
    unittest.main(verbosity=1)