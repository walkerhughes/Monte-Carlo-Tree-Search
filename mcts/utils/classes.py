import math
import random
import numpy as np

from ..settings.constants import MAX_CHILDREN, SEED_ANSWERS

class Node:
    def __init__(self, question, answer, parent=None):
        self.__dict__.update(locals())
        self.children = []
        self.visits = 0
        self.value = 0.0

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_fully_expanded(self):
        return len(self.children) >= MAX_CHILDREN

    def most_visited_child(self):
        return max(self.children, key = lambda child: child.visits)

    def best_child(self, exploration_weight=1.41):
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                weight = np.inf
            else:
                weight = (child.value / child.visits) + exploration_weight
            choices_weights.append(weight)
        return self.children[np.argmax(choices_weights)]
    

class MCTS:
    def __init__(self, question, seed_answers, iterations=2):
        self.__dict__.update(locals())
        self.root = Node(question, random.choice(SEED_ANSWERS))

    def search(self):
        for i in range(self.iterations):
            print(f"Iteration {i + 1}/{self.iterations}")
            node = self.select(self.root)
            print(f"Selected Node: {node.answer}")
            if not node.is_fully_expanded():
                # DEFINE: self.expand
                node = self.expand(node)
                print(f"Expanded node: {node.answer}")
            # DEFINE: self.simulate
            reward = self.simulate(node)
            print(f"Simulated Reward: {reward}")
            # DEFINE 
            self.backpropagate(node, reward)
        print(f"Visits to most visited child: {self.root.most_visited_child()}")
        return self.root.most_visited_child().answer
    
    def select(self, node):
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        return node
    
    def expand(self, node):
        for i in range(MAX_CHILDREN - len(node.children)):
            child_node = Node(self.question, node.answer, parent=node)
            node.add_child(child_node)

            critique = get_critique()