# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):  # repeat value iteration `iterations` times
            new_values = util.Counter()  # create a dictionary to store updated values for each state
            for state in self.mdp.getStates():  # for each state in the MDP
                if self.mdp.isTerminal(state):  # if the state is terminal, skip it
                    continue
                # Otherwise, update the value of the state using the Bellman equation
                new_values[state] = max(self.getQValue(state, action) for action in self.mdp.getPossibleActions(state))
            self.values = new_values.copy()  # replace the old values with the new values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***" 
         # using the formula: Q(s, a) = sum(T(s, a, s') * [R(s, a, s') + gamma * V(s')])
        # where T(s, a, s') is the probability of transitioning to state s' when taking action a in state s, 
        #  R(s, a, s') is the immediate reward of transitioning from state s to state s' by taking action a,
        # gamma is the discount factor, and V(s') is the value of the next state.
        return sum(prob * (self.mdp.getReward(state, action, new_state) + self.discount * self.values[new_state])
               for new_state, prob in self.mdp.getTransitionStatesAndProbs(state, action))


    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit. Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        # Get all possible actions for the given state
        actions = self.mdp.getPossibleActions(state)
        # If there are no legal actions (i.e. terminal state), return None
        if not actions:
            return None
        # For each possible action, compute the expected reward
        # summing over the probabilities of the possible next states weighted by their values, and store it in a dictionary
        action_values = {}
        for action in actions:
            state_prob = self.mdp.getTransitionStatesAndProbs(state, action)
            reward = sum(prob * (self.mdp.getReward(state, action, new_state) + \
                                 self.discount * self.values[new_state])
                         for new_state, prob in state_prob)
            action_values[action] = reward
        
        # return the action with the highest expected reward
        best_action = max(action_values, key=action_values.get)
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
