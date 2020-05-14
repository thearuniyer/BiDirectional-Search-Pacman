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

    # This implementation expands more nodes than the second one but it is accepted by the
    # autograder
    #print "Using DFS"

    # Set of candidate nodes to be explored
    # It will also hold the actions that were taken to get from the starting state to that state
    fringe = util.Stack()

    # List to keep track of what nodes (states) have been visited so as to not visit them again
    visited_nodes = []

    # Push in the first node along with a list of no actions
    # Since it is the starting node and no actions were taken to get to it
    # It will be popped first
    fringe.push((problem.getStartState(),[]))

    # Do this as long as the fringe is not empty
    while(fringe.isEmpty() == 0):
        # Pop the latest node and the actions taken to get to it
        working_state, actions_taken = fringe.pop()

        # As long as the state has not been visited ... 
        if(working_state not in visited_nodes):
            
            # Mark the state as visited
            visited_nodes.append(working_state)

            # If goal was found, return the actions taken to get to it
            if(problem.isGoalState(working_state)):
                return actions_taken
            
            # Add successors to the fringe
            for next_state in problem.getSuccessors(working_state):
                state, current_action, cost = next_state
                fringe.push((state,actions_taken + [current_action]))

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    #Search the shallowest nodes in the search tree first.
    "*** YOUR CODE HERE ***"
    # This implementation is basically the same as the one for DFS
    # only that a queque is used instead of a stack
    #print "Using BFS"
    fringe = util.Queue()
    visited_nodes = []

    fringe.push((problem.getStartState(),[]))

    while(fringe.isEmpty() == 0):
        working_state, actions_taken = fringe.pop()

        if(working_state not in visited_nodes):
            visited_nodes.append(working_state)

            if(problem.isGoalState(working_state)):
                return actions_taken

            for next_state in problem.getSuccessors(working_state):
                state, current_action, cost = next_state
                fringe.push((state,actions_taken + [current_action]))

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    #print "Using UCS"

    # Set of candidate nodes to be explored
    # It also holds actions needed to get there from the starting state
    # Priority queue is used to pop out the node with least cost
    fringe = util.PriorityQueue()

    # List of visited nodes
    visited_nodes = []

    # Push starting node and empty list of actions to queue, it has priority 0
    fringe.push( (problem.getStartState(), []),0)
    
    # Do this as long as the fringe is not empty
    while(fringe.isEmpty() == 0):
        # Pop cheapest node
        working_state, actions_taken = fringe.pop()

        # As long as node has not been added to the visited_nodes list, add it
        if(working_state not in visited_nodes):
            visited_nodes.append(working_state)
            
            # If the popped node is the solution return the actions taken to get to it
            if(problem.isGoalState(working_state)):
                #print "Actions taken to get from", problem.getStartState(), "to", state, "are", actions_taken + [current_action] #???
                return actions_taken

            # Push successors to the fringe and update it so the next node popped is the cheapest one
            for next_state in problem.getSuccessors(working_state):
                state, current_action, cost = next_state
                # If item is not in fringe, add it
                # If item is alredy in fringe but with higher priority, more cost?, update it with this one
                # If item is already in fringe but with less priority, cheaper?, do nothing
                # The predifined update utility function takes care of all these cases
                fringe.update((state, actions_taken + [current_action]), problem.getCostOfActions(actions_taken + [current_action]))

    # Implementation below "works" but does not pass autograder
    util.raiseNotDefined()

def nullHeuristic(state, problem=None, search_type=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    #print "Using A*"
    # This will be similar to uniform cost search
    # In this case the priority is the sum of the cost to the node + the heuristic

    fringe = util.PriorityQueue()
    visited_nodes = []

    # Push starting node to fringe 
    # It's priority is the heuristic we passed
    fringe.push( (problem.getStartState(), []), heuristic(problem.getStartState(),problem) )

    # While fringe is not empty
    while(fringe.isEmpty() == 0):
        #print("The priority is", fringe.peek_pr())
        working_state, actions_taken = fringe.pop()

        if (working_state not in visited_nodes):
            visited_nodes.append(working_state)

            if(problem.isGoalState(working_state)):
                return actions_taken

            for next_state in problem.getSuccessors(working_state):
                state, current_action, cost = next_state
                #gn = problem.getCostOfActions(actions_taken + [current_action])
                #hn = heuristic(working_state, problem)
                #print "gn =", gn,"hn =", hn
                fringe.update((state, actions_taken + [current_action]), problem.getCostOfActions(actions_taken+[current_action]) + heuristic(state,problem) )


    # This implementation is working but the autograder is not accpepting
    util.raiseNotDefined()

def bidirectionalSearch(problem, heuristic=nullHeuristic):

    # Auxiliary functions
    # This function will return the node with minimum priority and minumum g(n) value
    def getNode(open_list, minpr):
        # open_list contain tuples like the following
        # prFn/prBn, state, actions, g(n), f(n)

        # Must return node with priority == minpr and minimum g(n)

        ret_list = []
        for node in open_list:
            if (node[0] == minpr):
                ret_list.append(node)

        if(len(ret_list) == 1):
            return ret_list[0]

        elif(len(ret_list) > 1):
            ming = float("inf")
            ret_node = 0
            for node in ret_list:
                if (node[3] < ming):
                    ming = node
                    ret_node = node

            return ret_node

    # This function will return True if the state specified is present in the list passed
    def inList(open_list, state):
        # open_list contain tuples like the following
        # prFn/prBn, state, actions, g(n), f(n)
        for node in open_list:
            if node[1] == state:
                return (True, node)
        else:
            return (False, None)

    # This function will remove nodes with state "state" from open_list
    def removeState(open_list, state):
        # open_list contain tuples like the following
        # prFn/prBn, state,actions, g(n), f(n)

        for node in open_list:
            if(node[1] == state):
                open_list.remove(node)
            
    # This function returns prmin, fmin, gmin from the given list
    def getMin(open_list):
        # open_list contain tuples like the following
        # prFn/prBn, state,actions, g(n), f(n)

        prmin = float("inf")
        gmin = float("inf")
        fmin = float("inf")

        for node in open_list:
            nodepr = node[0]
            
            nodeg = node[3]
            nodef = node[4]

            if nodepr < prmin:
                prmin = nodepr

            if nodeg < gmin:
                gmin = nodeg

            if nodef < fmin:
                fmin = nodef
        
        return (prmin, gmin, fmin)

    # This function will take in a list of actions, mirror them and then reverse its order
    # Eg: East <-> West, North <-> South
    def reverseActions(action_list):
        mirrored = []
        for action in action_list:
            if (action == 'North'):
                mirrored.append('South')
            elif (action == 'South'):
                mirrored.append('North')
            elif (action == 'West'):
                mirrored.append('East')
            elif (action == 'East'):
                mirrored.append('West')

        mirrored.reverse()
        return mirrored

    # For each element added to the list we need:
    # priority
    # position
    # actions taken
    # g(n) = path cost
    # f(n) = g(n) + h(n) (h(n) = heuristic)

    # We also need to know at any given moment what the minimum priority at OpenF and OpenB is. 
    # As well as what the minimum f(n) and g(n) value in OpenF and Open B are

    # Lets keep global variables for
    # prminF, prminB, fminF, fminB, gminF and gminB and updated them accordingly

    # Global values that will be update it as necessary
    prminF = 0          # Minimum priority in OpenF
    prminB = 0          # Minimum priority in OpenB
    fminF = 0           # Minimum f value in OpenF
    fminB = 0           # Minimum f value in OpenB
    gminF = 0           # Minimum g (path cost) value in OpenF
    gminB = 0           # Minimum g (path cost) value in OpenB

    # Define Open lists, OpenF and OpenB, for fordward and backward search
    # These list will contain tuples in the form:
    # prFn/prBn, state, actions, g(n), f(n))
    OpenF = []
    OpenB = []

    start = problem.getStartState()
    goal = problem.goal

    # f and g values for node to be added to OpenF
    # gFn = problem.getCostOfActions([])
    gFn = 0
    fFn = gFn + heuristic(start, problem, start)

    # TODO - Define a manhattanHeuristic that searches for start rather than goal 
    # f and g values for node to be added to OpenB
    # gBn = problem.getCostOfActions([])
    gBn = 0
    fBn = gBn + heuristic(goal, problem, goal)

    # Priority for nodes to be pushed in OpenF and OpenB
    prFn = max(fFn, 2*gFn)
    prBn = max(fBn, 2*gBn)

    # Push start state to OpenF with priority prFn
    OpenF.append((prFn, start, [], gFn, fFn))

    # Push start state to OpenB with priority prBn
    OpenB.append((prBn, goal, [], gBn, fBn))

    # Define Closed lists for forward and backward search
    ClosedF = []
    ClosedB = []

    # U is the cost of the cheapest solution found so far. It is updated when a better solution is found
    U = float("inf")

    # epsilon is the cost of the cheapest edge in the state space. The cost of all edges is one, so the cheapest is 1
    epsilon = 1

    # Update min values
    prminF = prFn
    prminB = prBn
    fminF = fFn
    fminB = fBn
    gminF = gFn
    gminB = gBn

    # Posible list of actions to return
    perform_actions = []

    # Do this while neither of the open lists are empty
    while((len(OpenF)) != 0 and (len(OpenB) != 0)):
        # C min is the minimum of the minumum priorities of OpenF and OpenB.
        # Each iteration MM will expand a node with priority C
        C = min(prminF, prminB)

        if(U <= max(C, fminF, fminB, gminF+gminB + epsilon)):
            # Return the latest perform_actions 
            return perform_actions

        # Expand in the forward direction
        if(C == prminF):
            # Choose a node n in OpenF for which prFn == prminF and gFn is minimum.
            n = getNode(OpenF, prminF)

            prFn, nodeN, actions_taken, gFn, fFn = n

            # Move n from OpenF to ClosedF (Remove from OpenF and add to ClosedF)
            OpenF.remove(n)
            ClosedF.append(nodeN)

            for childC in problem.getSuccessors(nodeN):
                # childC comes in the format (location, action_taken, cost)
                state, current_action, cost = childC

                # get f and g values
                gFc = problem.getCostOfActions(actions_taken + [current_action])
                fFc = gFc + heuristic(state, problem, start)

                # If childC (state) is in OpenF or ClosedF and gFc <= gFn + cost(n,c) 
                # (cost to get from nodeN to childC which in this case will always be 1)
                if((state in ClosedF or inList(OpenF, state)[0]) and (gFc <= gFn + 1)):
                    # skip all the remaining loop contents and go back to the beginning of the loop
                    continue
                    
                if(state in ClosedF or inList(OpenF, state)[0]):
                    # remove c from OpenF or ClosedF

                    if(state in ClosedF):
                        ClosedF.remove(state)

                    if(inList(OpenF, state)[0]):
                        removeState(OpenF, state)

                # cost(n,c) is always 1 in this domain
                gFc = gFn + 1
                fFc = gFc + heuristic(state, problem, start)
                prFc = max(fFc, 2*gFc)

                # Add childC to OpenF
                # prFn/prBn, state,actions, g(n), f(n)
                fFn = gFn + heuristic(start, problem, start)
                OpenF.append((prFc, state, actions_taken + [current_action], gFc, fFc))

                # If childC is in OpenB then update U and perform_actions (the list of possible actions to perform)
                if(inList(OpenB, state)[0]):
                    prBc, state, actions_taken_Bsearch, gBc, fBc = inList(OpenB, state)[1]

                    actions_taken = actions_taken + [current_action]
                    backward_actions = reverseActions(actions_taken_Bsearch)
                    perform_actions = actions_taken +  backward_actions

                    U = min(U, gFc + gBc)

        # Expand in backward direction
        else:
            # Choose a node n in OpenB for which prBn == prminB and gBn is minimum.
            n = getNode(OpenB, prminB)

            prBn, nodeN, actions_taken, gBn, fBn = n

            # Move n from OpenB to ClosedB (Remove from OpenB and add to ClosedB)
            OpenB.remove(n)
            ClosedB.append(nodeN)

            for childC in problem.getSuccessors(nodeN):
                # childC comes in the format (location, action_taken, cost)
                state, current_action, cost = childC

                # get f and g values
                gBc = problem.getCostOfActionsBackward(actions_taken + [current_action])
                fBc = gBc + heuristic(state, problem, goal)

                # If childC (state) is in OpenB or ClosedB and gBc <= gBn + cost(n,c) 
                # (cost to get from nodeN to childC which in this case will always be 1)
                if((state in ClosedB or inList(OpenB, state)[0]) and (gBc <= gBn + 1)):
                    # skip all the remaining loop contents and go back to the beginning of the loop
                    continue
                
                if(state in ClosedB or inList(OpenB, state)[0]):
                    # remove c from OpenB or ClosedB
                    if(state in ClosedB):
                        ClosedB.remove(state)

                    if(inList(OpenB, state)[0]):
                        removeState(OpenB, state)

                # cost(n,c) is always 1 in this domain
                gBc = gBn + 1
                fBc = gBc + heuristic(state, problem, goal)

                prBc = max(fBc, 2*gBc)

                # Add childC to OpenF
                # prFn/prBn, state, actions, g(n), f(n)
                fBn = gBn + heuristic(start, problem, goal)
                OpenB.append((prBc, state, actions_taken + [current_action], gBc, fBc))

                 # If childC is in Openf then update U and perform_actions (the list of possible actions to perform)
                if(inList(OpenF, state)[0]):
                    prFc, state, actions_taken_Fsearch, gFc, fFc = inList(OpenF, state)[1]

                    actions_taken = actions_taken + [current_action]
                    backward_actions = reverseActions(actions_taken)
                    perform_actions = actions_taken_Fsearch +  backward_actions

                    U = min(U, gBc + gFc)

        # Update min values: prminF/B, gminF/B, fminF/B
        prminF, gminF, fminF = getMin(OpenF)
        prminB, gminB, fminB = getMin(OpenB)

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
bds = bidirectionalSearch
