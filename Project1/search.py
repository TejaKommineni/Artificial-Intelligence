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
import searchAgents


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

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    ans = []
    parent_child = {}
    visited = set()   
    stack = util.Stack()
    stack.push((problem.getStartState(), 'root', 0.0))    
    state = ()
    while not stack.isEmpty():
        state = stack.pop()
        visited.add(state[0])
        if problem.isGoalState(state[0]):            
            break
        for temp_state in  problem.getSuccessors(state[0]):
            if temp_state[0] in visited:
                continue
            else:
                parent_child[temp_state] = state
                stack.push(temp_state)
    root = state
    while root != (problem.getStartState(), 'root', 0.0):
        ans.append(root[1])
        root = parent_child[root]   
    ans.reverse()
    print ans
    return ans    
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    #python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs --frameTime 0
    #python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5 --frameTime 0
    "*** YOUR CODE HERE ***"
    q = util.Queue()
    root = (problem.getStartState(), None, None)
    q.push(root)
    #Keep track of visited nodes
    visited = util.Counter()
    #Dictionary that gives parent, given its children.
    parentDict = {root: None}
    #Did we find goal?
    bFound = False 
    while not q.isEmpty():
        top = q.pop()
        topStateCord = top[0]
        if visited[topStateCord] != 0:
            continue
        #mark it as visited
        visited[topStateCord] += 1
        if problem.isGoalState(topStateCord):
            bFound = True
            break
        #Insert all its unvisited successors in the q
        successors = problem.getSuccessors(topStateCord)
        for successor in successors:
            if visited[successor[0]] == 0:
                q.push(successor)
                parentDict[successor] = top
    path = []
    if bFound:
        #topState is the Goal.
        while top != root:
            path.insert(0, top[1])
            top = parentDict[top]
    return path 
    

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    ans = []
    parent_child = {}
    visited = set()   
    queue = util.PriorityQueue()
    queue.push((problem.getStartState(), 'root', 0.0),0)    
    state = ()
    while not queue.isEmpty():
        state = queue.pop()
        if state[0] in visited:
            continue
        visited.add(state[0])
        if problem.isGoalState(state[0]):            
            break
        for temp_state in  problem.getSuccessors(state[0]):
            if temp_state[0] in visited:
                continue
            else:
                lst = list(temp_state)
                lst[2] = lst[2]+state[2]
                temp_state = tuple(lst)
                parent_child[temp_state] = state
                queue.push(temp_state,temp_state[2])
    root = state
    while root != (problem.getStartState(), 'root', 0.0):
        ans.append(root[1])
        root = parent_child[root]   
    ans.reverse()
    print ans
    return ans
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    ans = []
    parent_child = {}
    visited = set()   
    queue = util.PriorityQueue()
    queue.push((problem.getStartState(), 'root', 0.0),0)    
    state = ()
    while not queue.isEmpty():
        state = queue.pop()
        if state[0] in visited:
            continue
        visited.add(state[0])
        if problem.isGoalState(state[0]):            
            break
        for temp_state in  problem.getSuccessors(state[0]):
            if temp_state[0] in visited:
                continue
            else:
                lst = list(temp_state)
                lst[2] = lst[2]+state[2]
                temp_state = tuple(lst)
                parent_child[temp_state] = state
                heur = heuristic(temp_state[0],problem)
                queue.push(temp_state,temp_state[2]+heur)
    root = state
    while root != (problem.getStartState(), 'root', 0.0):
        ans.append(root[1])
        root = parent_child[root]   
    ans.reverse()
    print ans
    return ans
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
