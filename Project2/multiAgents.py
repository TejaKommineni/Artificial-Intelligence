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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        print "legal moves", legalMoves
        print "scores", scores
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        ##FIXME: Pacman stops remove this dirty fix to avoid pacman stops issue!
        #if chosenIndex == 1: #Index of Stop action
        #    chosenIndex = random.choice(range(len(legalMoves)))

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        def euclidean_dist(pos1, pos2):
            return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


        next_ghost = 0
        for ghostIndex in range(len(newGhostStates)):
            ghostState = newGhostStates[ghostIndex]
            ghostPosition = ghostState.configuration.getPosition()
            if newPos == ghostPosition:
                next_ghost += 1
            print "ghost ", ghostIndex, ": Position - ", ghostPosition
        if next_ghost:
            return 0
        score = 1.0
        #print "newGhostStates:", newGhostStates
        #print "newScaredTimes:", newScaredTimes
        #print "Pacman Position:", successorGameState.getPacmanPosition()
        #print "action:", action
        newPos = [float(newPos[i]) for i in range(len(newPos))]
        #print "newPos:", newPos
        #TODO: why agent stops? DONE:
        dist = float('inf')
        for food in newFood.asList():
            dist = min(dist, euclidean_dist(food, newPos))
        foodCount = newFood.count()
        ghost_dist = float('inf')
        for ghostPos in successorGameState.getGhostPositions():
            ghost_dist = min(ghost_dist, manhattanDistance(ghostPos, newPos))

        #print "dist to food:", dist
        #print "score:", successorGameState.getScore()
        if ghost_dist < 4.0:
            score = (ghost_dist ** 5) + (100.0 / float(dist)) ** 2
        else:
            score = (1.0/float(dist)) ** 2 + (10000.0/float(foodCount + 1)) ** 2

        #score = (ghost_dist ** 10)/10.0 + (100.0 / float(dist))
        return score

def scoreEvaluationFunction(currentGameState):
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
        self.pacmanIndex = 0
        
    def isTerminal(self, state, depth, agent):
        return depth == self.depth or \
               state.isWin() or \
               state.isLose() or \
               state.getLegalActions(agent) == 0

    # is this agent pacman
    def isPacman(self, state, agent):
        return agent % state.getNumAgents() == 0

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        depth = 0
        agent = 0
        val = self.dispatch(gameState, agent, depth)        
        return val[0]

    def dispatch(self, gameState, agent, depth): 
        # if both pacman and ghosts are done at this depth we increase the depth and start with pacman.
        if agent >= gameState.getNumAgents():
            agent = 0
            depth = depth + 1
        
        # if the current depth the max depth specified we call the evaluation function.
        if depth == self.depth:
            return self.evaluationFunction(gameState)
        
        if  agent == self.pacmanIndex:
            return self.max_value(gameState, agent, depth)
        else:
            return self.min_value(gameState, agent, depth)
        
    def max_value(self, gameState, agent, depth):
        val = ("unknown", -1*float("inf"))
        
        # if there are no more legal states we call the evaluation function.
        if not gameState.getLegalActions(agent):
            return self.evaluationFunction(gameState)

        # else in all the other actions we find the minimum value.
        for action in gameState.getLegalActions(agent):
            if action == "Stop":
                continue
            
            ret_val = self.dispatch(gameState.generateSuccessor(agent, action), agent + 1, depth)
            if type(ret_val) is tuple:
                ret_val = ret_val[1] 

            val_new = max(val[1], ret_val)
            
            if val_new is not val[1]:
                val = (action, val_new) 
              
        return val
    
        
    def min_value(self, gameState, agent, depth):
        val = ("unknown", float("inf"))
        
        # if there are no more legal states we call the evaluation function.
        if not gameState.getLegalActions(agent):
            return self.evaluationFunction(gameState)
        
        # else in all the actions we find the minimum value.
        for action in gameState.getLegalActions(agent):
            if action == "Stop":
                continue
            
            ret_val = self.dispatch(gameState.generateSuccessor(agent, action), agent + 1, depth)
            if type(ret_val) is tuple:
                ret_val = ret_val[1] 

            val_new = min(val[1], ret_val)

            if val_new is not val[1]:    
                val = (action, val_new) 
            
        return val
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
   
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """               
        _,action = self.dispatch(gameState, 0, 0)        
        return action

    def dispatch(self, gameState, depth, agent, A=float("-inf"), B=float("inf")):
            # If all the agents explored this depth.
            if agent == gameState.getNumAgents():  
                depth += 1
                agent = 0

            # Determining terminal states.
            if depth == self.depth or gameState.isWin() or gameState.isLose() or gameState.getLegalActions(agent) == 0:
                return self.evaluationFunction(gameState), None

            if self.isPacman(gameState, agent):
                return self.get_value(gameState, depth, agent, A, B, float('-inf'), max)
            else:
                return self.get_value(gameState, depth, agent, A, B, float('inf'), min)
            
 

    def get_value(self, state, depth, agent, A, B, scr, fn):
            best_score = scr
            best_action = None

            for action in state.getLegalActions(agent):
                score,_ = self.dispatch(state.generateSuccessor(agent, action), depth, agent + 1, A, B)
                best_score, best_action = fn((best_score, best_action), (score, action))

                if self.isPacman(state, agent):
                    if best_score > B:
                        return best_score, best_action
                    A = fn(A, best_score)
                else:
                    if best_score < A:
                        return best_score, best_action
                    B = fn(B, best_score)

            return best_score, best_action
        
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
        
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        selected_action = {}
        for action in  gameState.getLegalActions(0):
           selected_action[action] = self.expectimax(gameState.generateSuccessor(0, action), 0, 1)            
        
        return max(selected_action, key=selected_action.get) 
    
    def expectimax(self, state, depth, agent):
            #if both pacman and agents played in this depth.
            if agent == state.getNumAgents():  
                return self.expectimax(state, depth + 1, 0)  # increasing the depth

            # Determining terminal states.
            if depth == self.depth or state.isWin() or state.isLose() or state.getLegalActions(agent) == 0:
                return self.evaluationFunction(state)  # Terminal states are sent to evaluation function.
            
            #if we have other ghosts to make their move
            successors = [
                self.expectimax(state.generateSuccessor(agent, action), depth, agent + 1)
                for action in state.getLegalActions(agent)
            ]

            # checking for pacman.
            if agent % state.getNumAgents() == 0:
                return max(successors)
            # finding the expected value for ghosts.
            else:
                return sum(successors)/len(successors)

class dp_calculator():
    def __init__(self):
        self.test = True
    def shortestMatrix(self, matrix, walls):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        ans = [[float('inf') for j in range(matrix.height)] for i in range(matrix.width)]
        #print ans
        
        #base cases:
        prev = float('inf')
        for j in range(matrix.height - 1, -1, -1):
            #if matrix[-1][j] != False:
            if matrix[-1][j] != False:
                if not walls[-1][j]:
                    ans[-1][j] = min(ans[-1][j], prev + 1)
            else:
                ans[-1][j] = 0
            prev = ans[-1][j]
        prev = float('inf')
        for i in range(matrix.width - 1, -1, -1):
            if matrix[i][-1] != False:
                if not walls[i][-1]:
                    ans[i][-1] = min(ans[i][-1], prev + 1)
            else:
                ans[i][-1] = 0
            prev = ans[i][-1]
        for i in range(matrix.width - 2, -1, -1):
            for j in range(matrix.height - 2, -1, -1):
                if matrix[i][j] == False:
                    ans[i][j] = 0
                else:
                    if not walls[i][j]:
                        ans[i][j] = min(ans[i][j], ans[i][j + 1] + 1, ans[i + 1][j] + 1)
        #base case 2
        prev = float('inf')
        for j in range(matrix.height):
            if matrix[0][j] != False:
                if not walls[0][j]:
                    ans[0][j] = min(ans[0][j], prev + 1)
            else:
                ans[0][j] = 0
            prev = ans[0][j]
        prev = float('inf')
        for i in range(matrix.width):
            if matrix[i][0] != False:
                if not walls[i][0]:
                    ans[i][0] = min(ans[i][0], prev + 1)
            else:
                ans[i][0] = 0
            prev = ans[i][0]
        for i in range(1, matrix.width, 1):
            for j in range(1, matrix.height, 1):
                if not walls[i][j]:
                    ans[i][j] = min(ans[i][j], ans[i][j - 1] + 1, ans[i - 1][j] + 1)
        return ans

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newWalls = currentGameState.getWalls()
    capsules = currentGameState.getCapsules()
    #print "scared times: ", newScaredTimes
    
    def euclidean_dist(pos1, pos2):
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


    next_ghost = 0
    for ghostIndex in range(len(newGhostStates)):
        ghostState = newGhostStates[ghostIndex]
        ghostPosition = ghostState.configuration.getPosition()
        if newPos == ghostPosition:
            next_ghost += 1
        print "ghost ", ghostIndex, ": Position - ", ghostPosition
    #if sum(newScaredTimes) != 0 and next_ghost:
        #return 10000000000000
    if next_ghost:
        return 0
    score = 1.0
    #print "newGhostStates:", newGhostStates
    #print "newScaredTimes:", newScaredTimes
    #print "Pacman Position:", successorGameState.getPacmanPosition()
    #print "action:", action
    newPos = [float(newPos[i]) for i in range(len(newPos))]
    #print "newPos:", newPos
    #TODO: why agent stops? DONE:
    dist = float('inf')
    for food in newFood.asList():
        dist = min(dist, euclidean_dist(food, newPos))
    dist_capsules = float('inf')
    for cap in capsules:
        dist_capsules = min(dist_capsules, euclidean_dist(cap, newPos))
    ###############
    dp = dp_calculator()
    #dist_mat = dp.shortestMatrix(newFood, newWalls)
    #dist = dist_mat[int(newPos[0] + 0.5)][int(newPos[1] + 0.5)] + 1
    ###############
    foodCount = newFood.count()
    ghost_dist = float('inf')
    for ghostPos in currentGameState.getGhostPositions():
        ghost_dist = min(ghost_dist, manhattanDistance(ghostPos, newPos))

    #print "dist to food:", dist
    if ghost_dist < 4.0 and sum(newScaredTimes) == 0:
        score = (ghost_dist ** 5) + (100.0 / float(dist)) ** 2
    else:
        score = (1.0/float(dist)) ** 2 + (10000.0/float(foodCount + 1)) ** 2 + 10000.0/(len(capsules) + 1) ** 4

    #score = (ghost_dist ** 10)/10.0 + (100.0 / float(dist))
    return score

# Abbreviation
better = betterEvaluationFunction