import math
import random
import numpy as np
from openai import OpenAI

from utils.funcs import *
from settings.constants import MAX_CHILDREN, SEED_ANSWERS, OPENAI_MODEL

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

    def best_child(self, exploration_weight=math.sqrt(2)):
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                weight = np.inf
            else:
                weight = (child.value / child.visits) + exploration_weight * math.sqrt(2 * math.log(self.value) / self.visits)
            choices_weights.append(weight)
        return self.children[np.argmax(choices_weights)]
    

class MCTS:
    def __init__(self, client: OpenAI, question: str, seed_answers: list, iterations: int = 2):
        self.__dict__.update(locals())
        self.root = Node(question, random.choice(SEED_ANSWERS))

    def search(self):
        for i in range(self.iterations):
            print(f"Iteration {i + 1}/{self.iterations}")
            node = self.select(self.root)
            print(f"Selected Node: {node.answer}")
            if not node.is_fully_expanded():
                node = self.expand(node)
                print(f"Expanded node: {node.answer}")
            reward = self.simulate(node)
            print(f"Simulated Reward: {reward}")
            self.backpropagate(node, reward)
        print(f"Visits to most visited child: {self.root.most_visited_child()}")
        return self.root.most_visited_child().answer
    
    def select(self, node):
        """ 
        For a fully expanded Node, move to its best child using UCT criteria.
        """
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        return node
    
    def expand(self, node):
        for i in range(MAX_CHILDREN - len(node.children)):

            child_node = Node(self.question, node.answer, parent=node)
            node.add_child(child_node)

            critique = get_draft_answer_critique(self.client, OPENAI_MODEL, self.question, child_node.answer)
            print(f"\n---Critique {i}---\n{critique}")

            improved_answer = get_improved_answer(self.client, OPENAI_MODEL, self.question, child_node.answer, critique)
            print(f"\n---Improved Answer {i}---\n{improved_answer}")

            child_node.answer = improved_answer
        return random.choice(node.children)
    
    def simulate(self, node):
        return get_answer_rating(self.client, OPENAI_MODEL, node.question, node.answer)
    
    def backpropagate(self, node, reward):
         """ 
         Backpropagates reward for a given answer through the tree structure,
         which is used to calculate UTC bounds for the best answer to the original question.
         """
         while node is not None: 
             node.visits += 1
             node.value += reward
             node = node.parent
