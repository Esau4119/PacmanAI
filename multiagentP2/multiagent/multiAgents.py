# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()


        "*** YOUR CODE HERE ***"
        # Compute nearest ghost distance
        ghost_distances = [manhattanDistance(ghost.getPosition(), newPos)
                           for ghost in newGhostStates if ghost.scaredTimer == 0]
        if ghost_distances:
            nearest_ghost_dis = min(ghost_distances)
        else:
            # If there are no ghosts, set nearest_ghost_dis to a high value
            nearest_ghost_dis = 1e9

        # Compute nearest food distance
        food_distances = [manhattanDistance(food, newPos) for food in newFood.asList()]
        if food_distances:
            nearest_food_dis = min(food_distances)
        else:
            # If there is no food left, set nearest_food_dis to 0
            nearest_food_dis = 0

        # Combine the distances and current score to get the final evaluation
        return successorGameState.getScore() - 7 / (nearest_ghost_dis + 1)\
               - nearest_food_dis / 3


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimaxSearch(gameState, agentIndex=0, depth=self.depth)[1]
    def minimaxSearch(self, gameState, agentIndex, depth):
    # check if we've reached the specified depth or if we've won/lost the game
        if depth == 0 or gameState.isLose() or gameState.isWin():
            # if so, return the evaluation function value and a default stop action
            return self.evaluationFunction(gameState), Directions.STOP
        if agentIndex == 0:
            # if it's pacman's turn, call the maximizer function
            return self.maximizer(gameState, agentIndex, depth)
        else:
            # if it's a ghost's turn, call the minimizer function
            return self.minimizer(gameState, agentIndex, depth)

    def minimizer(self, gameState, agentIndex, depth):
        # get the legal actions for the current ghost
        actions = gameState.getLegalActions(agentIndex)
        # calculate the index of the next agent in the game (either the next ghost or pacman)
        next_agent = (agentIndex + 1) % gameState.getNumAgents()
        # if we've cycled through all the ghosts, we're one level deeper in the tree
        if next_agent == 0:
            depth -= 1
        # initialize the minimum score to infinity and the minimum action to a default stop action
        min_score = float('inf')
        min_action = Directions.STOP
        # loop through all the legal actions for the current ghost
        for action in actions:
            # generate the successor state for the current ghost taking the current action
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            # recursively call the minimax search function for the next agent in the game
            # and update the minimum score and action if necessary
            new_score, _ = self.minimaxSearch(successor_game_state, next_agent, depth)
            if new_score < min_score:
                min_score, min_action = new_score, action
        # return the minimum score and action
        return min_score, min_action

    def maximizer(self, gameState, agentIndex, depth):
        # get the legal actions for pacman
        actions = gameState.getLegalActions(agentIndex)
        # calculate the index of the next agent in the game (either the next ghost or pacman)
        next_agent = (agentIndex + 1) % gameState.getNumAgents()
        # if we've cycled through all the ghosts, we're one level deeper in the tree
        if next_agent == 0:
            depth -= 1
        # initialize the maximum score to negative infinity and the maximum action to a default stop action
        max_score = float('-inf')
        max_action = Directions.STOP
        # loop through all the legal actions for pacman
        for action in actions:
            # generate the successor state for pacman taking the current action
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            # recursively call the minimax search function for the next agent in the game
            # and update the maximum score and action if necessary
            new_score, _ = self.minimaxSearch(successor_game_state, next_agent, depth)
            if new_score > max_score:
                max_score, max_action = new_score, action
        # return the maximum score and action
        return max_score, max_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.alphabetaSearch(gameState, 0, self.depth, -float('inf'), float('inf'))[1]

    def alphabetaSearch(self, gameState, agentIndex, depth, alpha, beta):
        """
        Performs an alpha-beta search on the given game state, starting from the given agent index and depth,
        with the given alpha and beta values.
        """
        # Base case: reached maximum depth or game over
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        # Max player's turn
        elif agentIndex == 0:
            return self.maximize(gameState, agentIndex, depth, alpha, beta)
        # Min player's turn
        else:
            return self.minimize(gameState, agentIndex, depth, alpha, beta)

    def maximize(self, gameState, agentIndex, depth, alpha, beta):
        """
        Maximizes the score of the current agent, given the current game state, and returns the resulting score and action.
        """
        max_score, max_action = -float('inf'), Directions.STOP
        # Generate successor states and evaluate them recursively
        for action in gameState.getLegalActions(agentIndex):
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            new_score, _ = self.alphabetaSearch(successor_game_state, self.nextAgent(agentIndex, gameState), self.nextDepth(agentIndex, depth, gameState), alpha, beta)
            # Update maximum score and action
            if new_score > max_score:
                max_score, max_action = new_score, action
            # Perform alpha-beta pruning
            if new_score > beta:
                return new_score, action
            alpha = max(alpha, max_score)
        return max_score, max_action

    def minimize(self, gameState, agentIndex, depth, alpha, beta):
        """
        Minimizes the score of the current agent, given the current game state, and returns the resulting score and action.
        """
        min_score, min_action = float('inf'), Directions.STOP
        # Generate successor states and evaluate them recursively
        for action in gameState.getLegalActions(agentIndex):
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            new_score, _ = self.alphabetaSearch(successor_game_state, self.nextAgent(agentIndex, gameState), self.nextDepth(agentIndex, depth, gameState), alpha, beta)
            # Update minimum score and action
            if new_score < min_score:
                min_score, min_action = new_score, action
            # Perform alpha-beta pruning
            if new_score < alpha:
                return new_score, action
            beta = min(beta, min_score)
        return min_score, min_action

    def nextAgent(self, agentIndex, gameState):
        """
        Returns the index of the next agent to play, given the current agent index and game state.
        """
        return (agentIndex + 1) % gameState.getNumAgents()

    def nextDepth(self, agentIndex, depth, gameState):
        """
        Returns the remaining depth for the next agent to play, given the current agent index, current depth, and game state.
        """
        return depth - int(agentIndex == gameState.getNumAgents() - 1)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimaxSearch(gameState, 0, self.depth)[1]
    def expectimaxSearch(self, gameState, agentIndex, depth):
        """
        Recursively computes the expectimax search for the given game state and agent index
        """
        if depth == 0 or gameState.isWin() or gameState.isLose():
            # if we reach the bottom of the search tree or the game is over, return the score and stop
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            # if it's Pacman's turn, call the maximizer function
            return self.maximizer(gameState, agentIndex, depth)
        else:
            # otherwise, call the expectation function
            return self.expectation(gameState, agentIndex, depth)
    def maximizer(self, gameState, agentIndex, depth):
        """
        Computes the maximum score and corresponding action for the given game state and agent index
        """
        # get legal actions for the current agent
        actions = gameState.getLegalActions(agentIndex)
        # set the next agent and depth
        if agentIndex == gameState.getNumAgents() - 1:
            nextAgent, nextDepth = 0, depth - 1
        else:
            nextAgent, nextDepth = agentIndex + 1, depth
        # initialize maximum score and action
        maxScore, maxAction = -1e9, Directions.STOP
        # loop through legal actions and calculate the score for each successor game state
        for action in actions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            # call the expectimax search recursively with the next agent and depth
            newScore = self.expectimaxSearch(successorGameState, nextAgent, nextDepth)[0]
            # update maximum score and action
            if newScore > maxScore:
                maxScore, maxAction = newScore, action
        # return maximum score and corresponding action
        return maxScore, maxAction

    def expectation(self, gameState, agentIndex, depth):
        """
        Computes the expected score and a random action for the given game state and agent index
        """
        # get legal actions for the current agent
        actions = gameState.getLegalActions(agentIndex)
        # set the next agent and depth
        if agentIndex == gameState.getNumAgents() - 1:
            nextAgent, nextDepth = 0, depth - 1
        else:
            nextAgent, nextDepth = agentIndex + 1, depth
        # initialize expected score and action
        expScore, expAction = 0, Directions.STOP
        # loop through legal actions and calculate the score for each successor game state
        for action in actions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            # call the expectimax search recursively with the next agent and depth
            expScore += self.expectimaxSearch(successorGameState, nextAgent, nextDepth)[0]
        # compute the expected score by averaging over all successor game states
        expScore /= len(actions)
        # return expected score and a random action (not used in expectimax)
        return expScore, expAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    The function calculates this score based on several factors.
    First, it gets the current position of Pacman (pacman_pos), the positions of all the food dots on the board (foods), and the positions and scared times of all the ghosts (ghost_states and scared_times).
    It then calculates the distance between Pacman and the nearest ghost (nearest_ghost_dis) and the distance between Pacman and the nearest food dot (nearest_food_dis). 
    If Pacman is closer to a ghost, the score is penalized, and if Pacman is closer to a food dot, the score is rewarded. 
    The score is then adjusted by subtracting 7 divided by the distance to the nearest ghost plus 1, and dividing the distance to the nearest food dot by 3.

    If Pacman is close to a scared ghost, it omits the ghost's action for a while by setting the distance to the nearest scared ghost to -10. Finally, the function returns the current score of the game (current_game_state.getScore()) adjusted by the penalties and rewards calculated based on Pacman's position relative to the ghosts and food dots on the board.
    """
    "*** YOUR CODE HERE ***"
    pacman_pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghosts = currentGameState.getGhostStates()

    # Calculate the total number of food pellets remaining
    food_left = food.count()

    # Calculate the distance to the closest ghost and its state
    min_ghost_distance = float('inf')
    is_ghost_scared = False
    for ghost in ghosts:
        ghost_pos = ghost.getPosition()
        ghost_distance = manhattanDistance(pacman_pos, ghost_pos)
        if ghost_distance < min_ghost_distance:
            min_ghost_distance = ghost_distance
            is_ghost_scared = ghost.scaredTimer > 0

    # Calculate the distance to the closest food pellet
    if food_left > 0:
        min_food_distance = min(manhattanDistance(pacman_pos, food_pos) for food_pos in food.asList())
    else:
        min_food_distance = 0

    # Calculate the score of the current game state
    score = currentGameState.getScore()

    # Use the heuristics to evaluate the current game state
    if is_ghost_scared:
        ghost_weight = 0.5
    else:
        ghost_weight = 2.0
    return score - ghost_weight / (min_ghost_distance + 1) - min_food_distance / 3

# Abbreviation
better = betterEvaluationFunction
