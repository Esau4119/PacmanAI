# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

class Node:
    def __init__(self, state, pred, action, priority=0):
        self.state = state
        self.pred = pred
        self.action = action
        self.priority = priority
    def __repr__(self):
        return "State: {0}, Action: {1}".format(self.state, self.action)

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    "*** YOUR CODE HERE ***"   
    stack = util.Stack()
    stack.push(Node(problem.getStartState(), None, None))
    visited = set()
    actions = list()
    while not stack.isEmpty():
        node = stack.pop() # Pop the next node from the stack
        if problem.isGoalState(node.state): # required check to see if our current state is a  goal state
            return  isGoal(node,actions) #returns list of actions that we need to reach goal if we are on the goal state,
        if node.state not in visited:
            visited.add(node.state)
            for s in problem.getSuccessors(node.state):# grabbing our next node and pushing into stack
                stack.push(Node(s[0], node, s[1]))
    return list()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    stack = util.Queue()
    stack.push(Node(problem.getStartState(), None, None))
    visited = set()
    actions = list()
    while not stack.isEmpty():
        node = stack.pop() # Pop the next node from the stack
        if problem.isGoalState(node.state): # required check to see if our current state is a  goal state
            return  isGoal(node,actions) #returns list of actions that we need to reach goal if we are on the goal state,
        if node.state not in visited:
            visited.add(node.state)
            for s in problem.getSuccessors(node.state):# grabbing our next node and pushing into stack
                stack.push(Node(s[0], node, s[1]))
    return list()

def isGoal(node,actions):
            while node.action:
                actions.append(node.action)
                node = node.pred
            actions.reverse()
            return actions

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    stack = util.PriorityQueue() # now we can use a priority Queue
    stack.push(Node(problem.getStartState(), None, None),0)
    visited = set()
    actions = list()
    while not stack.isEmpty():
        node = stack.pop() # Pop the next node from the stack
        if problem.isGoalState(node.state): # required check to see if our current state is a  goal state
            return  isGoal(node,actions) #returns list of actions that we need to reach goal if we are on the goal state,
        if node.state not in visited:
            visited.add(node.state)
            for s in problem.getSuccessors(node.state):
                #  Add the successor node to the priority queue with a priority value that takes into 
               # account the cost so far and the estimated cost to the goal
                stack.push(Node(s[0], node, s[1], s[2]+node.priority), s[2]+node.priority)
    return list()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    stack = util.PriorityQueue()
    start = problem.getStartState()
    stack.push(Node(start, None, None,heuristic(start, problem)),heuristic(start, problem))
    visited = set()
    actions = list()
    while not stack.isEmpty():
        node = stack.pop() # Pop the next node from the stack
        if problem.isGoalState(node.state): # required check to see if our current state is a  goal state
            return  isGoal(node,actions) #returns list of actions that we need to reach goal if we are on the goal state,
        if node.state not in visited:
            visited.add(node.state)
            for s in problem.getSuccessors(node.state):
              #  node's successors to the stack with their costs and heuristics
                stack.push(Node(s[0], node, s[1], s[2]+node.priority), s[2]+node.priority+heuristic(s[0], problem))
    return list()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
