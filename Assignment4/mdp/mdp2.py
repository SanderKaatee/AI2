# code for representing/solving an MDP

import numpy
import time
from numpy.random import randint
from problem_utils import *


class State:

    def __init__(self):
        self.utility = 0.0
        self.reward = 0.0
        # an action maps to a list of probability/state pairs
        self.transitions = {}
        self.actions = []
        self.policy = None
        self.coords = 0
        self.isGoal = False
        self.isWall = False
        self.id = 0

    def computeEU(self, action):
        return sum([trans[0] * trans[1].utility
                    for trans in self.transitions[action]])

    def selectBestAction(self):
        best = max([(self.computeEU(a), a) for a in self.actions])
        return best[1]


class Map:
    def __init__(self):
        self.states = {}
        self.stop_crit = 0.0000001
        self.gamma = 0.5
        self.n_rows = 0
        self.n_cols = 0

    class PrintType:
        ACTIONS = 0
        VALUES = 1

    def valueIteration(self) :
        ### Declare intitial values
        epoch = 0
        maxChangeStates=None;

        ### 1. initialize utilities to 0
        for state in self.states.values():
           if state.isGoal == False :
             state.utility=0
        ### 2. repeat value iteration loop until largest change is smaller than
        while True:
          change = numpy.array([])
          for state in self.states.values():
            ### For every non-goal state compute the best action and it's utility
            if state.isGoal == False :
              oldUtility=state.utility
              utilBestAction = state.computeEU(state.selectBestAction())
              newUtility = state.reward + self.gamma *utilBestAction
              change = numpy.append(change, [oldUtility-newUtility])
              state.utility=newUtility
          ### Create an array with the changes of every state
          maxChangeStates=abs(max(change))
          ### Stop criterion
          epoch = epoch + 1
          if(self.stop_crit > maxChangeStates):
                print(epoch)
                break

    # you write this method
    def policyIteration(self):
        epoch = 0
        # 1. initialize random policy
        for s in self.states.values():
            # For each state, select a random action.
            selectedAction = randint(len(s.actions))
            s.policy = s.actions[selectedAction]

        # 2 repeat policy iteration loop until policy is stable
        changed = True
        while changed:
            epoch = epoch + 1
            changed = False
            # Do policy evaluation and update the utilities accordingly.
            self.calculateUtilitiesLinear()

            # Check if any state's policy can be optimized. If so, change that state's policy.
            for s in self.states.values():
                optimalAction = s.selectBestAction()
                if not s.policy == optimalAction:  # If the policy's action is not the optimal action,
                    s.policy = optimalAction  # then change the policy
                    changed = True
        print(epoch)


    def calculateUtilitiesLinear(self):
        n_states = len(self.states)
        coeffs = numpy.zeros((n_states, n_states))
        ordinate = numpy.zeros((n_states, 1))
        for s in self.states.values():
            row = s.id
            ordinate[row, 0] = s.reward
            coeffs[row, row] += 1.0
            if not s.isGoal:
                probs = s.transitions[s.policy]
                for p in probs:
                    col = p[1].id
                    coeffs[row, col] += -self.gamma * p[0]
        solution, _, _, _ = numpy.linalg.lstsq(coeffs, ordinate, rcond=None)
        for s in self.states.values():
            if not s.isGoal:
                s.utility = solution[s.id, 0]

    def printActions(self):
        self.printMaze(self.PrintType.ACTIONS)

    def printValues(self):
        self.printMaze(self.PrintType.VALUES)

    def printMaze(self, print_type):
        to_print = ":"
        for c in range(self.n_cols):
            to_print = to_print + "--------:"
        to_print = to_print + '\n'
        for r in range(self.n_rows):
            to_print = to_print + "|"
            for c in range(self.n_cols):
                if self.states[(c, r)].isWall:
                    to_print = to_print + "        "
                else:
                    to_print = to_print + ' '
                    if self.states[(c, r)].isGoal:
                        to_print = to_print + \
                                   "  {0: d}  ".format(int(self.states[(c, r)].utility))
                    else:
                        if print_type == self.PrintType.VALUES:
                            to_print = to_print + \
                                       "{0: .3f}".format(self.states[(c, r)].utility)
                        elif print_type == self.PrintType.ACTIONS:
                            a = self.states[(c, r)].policy  # Used to be: a = self.states[(c, r)].selectBestAction()
                            to_print = to_print + "  "
                            if a == 'left':
                                to_print = to_print + "<<"
                            elif a == 'right':
                                to_print = to_print + ">>"
                            elif a == 'up':
                                to_print = to_print + "/\\"
                            elif a == 'down':
                                to_print = to_print + "\\/"
                            to_print = to_print + "  "
                    to_print = to_print + ' '
                to_print = to_print + "|"
            to_print = to_print + '\n'
            to_print = to_print + ":"
            for c in range(self.n_cols):
                to_print = to_print + "--------:"
            to_print = to_print + '\n'
        print(to_print)


def makeRNProblem():
    """
    Creates the maze defined in Russell & Norvig. Utilizes functions defined
    in the problem_utils module.
    """

    walls = [(1, 1)]
    actions = ['left', 'right', 'up', 'down']
    cols = 4
    rows = 3

    def filterState(oldState, newState):
        if (newState[0] < 0 or newState[1] < 0 or newState[0] > cols - 1 or
                newState[1] > rows - 1 or newState in walls):
            return oldState
        else:
            return newState

    m = Map()
    m.n_cols = cols
    m.n_rows = rows
    for i in range(m.n_cols):
        for j in range(m.n_rows):
            m.states[(i, j)] = State()
            m.states[(i, j)].coords = (i, j)
            m.states[(i, j)].isGoal = False
            m.states[(i, j)].actions = actions
            m.states[(i, j)].id = j * m.n_cols + i
            m.states[(i, j)].reward = -0.04

    m.states[(3, 0)].isGoal = True
    m.states[(3, 1)].isGoal = True

    m.states[(3, 0)].utility = 1.0
    m.states[(3, 1)].utility = -1.0

    m.states[(3, 0)].reward = 1.0
    m.states[(3, 1)].reward = -1.0

    for t in walls:
        m.states[t].isGoal = True
        m.states[t].isWall = True
        m.states[t].reward = 0.0
        m.states[t].utility = 0.0

    for s in m.states.items():
        for a in actions:
            s[1].transitions[a] = [
                (0.8, m.states[filterState(s[0], getSuccessor(s[0], a))]),
                (0.1, m.states[filterState(s[0], getSuccessor(s[0], left(a)))]),
                (0.1, m.states[filterState(s[0], getSuccessor(s[0], right(a)))])]
    return m


def make2DProblem():
    """
    Creates the larger maze described in the exercise. Utilizes functions
    defined in the problem_utils module.
    """

    walls = [(1, 1), (4, 1), (5, 1), (6, 1), (7, 1), (1, 2), (7, 2), (1, 3), (5, 3),
             (7, 3), (1, 4), (5, 4), (7, 4), (1, 5), (5, 5), (7, 5), (1, 6), (5, 6),
             (7, 6), (1, 7), (5, 7), (7, 7), (1, 8), (3, 8), (4, 8), (5, 8),
             (7, 8), (1, 9)]
    actions = ['left', 'right', 'up', 'down']

    def filterState(oldState, newState):
        if (newState[0] < 0 or newState[1] < 0 or newState[0] > 9 or
                newState[1] > 9 or newState in walls):
            return oldState
        else:
            return newState

    m = Map()
    m.n_cols = 10
    m.n_rows = 10
    for i in range(m.n_cols):
        for j in range(m.n_rows):
            m.states[(i, j)] = State()
            m.states[(i, j)].coords = (i, j)
            m.states[(i, j)].isGoal = False
            m.states[(i, j)].actions = actions
            m.states[(i, j)].id = j * 10 + i
            m.states[(i, j)].reward = -0.04

    m.states[(0, 9)].isGoal = True
    m.states[(9, 9)].isGoal = True
    m.states[(9, 0)].isGoal = True

    m.states[(0, 9)].utility = 1.0
    m.states[(9, 9)].utility = -1.0
    m.states[(9, 0)].utility = 1.0

    m.states[(0, 9)].reward = 1.0
    m.states[(9, 9)].reward = -1.0
    m.states[(9, 0)].reward = 1.0

    for t in walls:
        m.states[t].isGoal = True
        m.states[t].isWall = True
        m.states[t].utility = 0.0
        m.states[t].reward = 0.0

    for s in m.states.items():
        for a in actions:
            s[1].transitions[a] = [
                (0.7, m.states[filterState(s[0], getSuccessor(s[0], a))]),
                (0.1, m.states[filterState(s[0], getSuccessor(s[0], opposite(a)))]),
                (0.1, m.states[filterState(s[0], getSuccessor(s[0], left(a)))]),
                (0.1, m.states[filterState(s[0], getSuccessor(s[0], right(a)))])]

    return m


def main():

    #RN value iteration
    print('RN value iteration')
    start = time.time()
    m1 = makeRNProblem()
    m1.valueIteration()
    m1.printValues()
    m1.printActions()
    end = time.time()
    print("time taken: ", end-start)
    #RN Policy iteration
    print('RN Policy iteration')
    start = time.time()
    m2 = makeRNProblem()
    m2.policyIteration()
    m2.printActions()
    end = time.time()
    print("time taken: ", end-start)
    #2D value iteration
    print('2D value iteration')
    start = time.time()
    m1_1 = make2DProblem()
    m1_1.valueIteration()
    m1_1.printValues()
    m1_1.printActions()
    end = time.time()
    print("time taken: ", end-start)
    #2D Policy iteration
    print('2D Policy iteration')
    start = time.time()
    m2_1 = make2DProblem()
    m2_1.policyIteration()
    m2_1.printActions()
    end = time.time()
    print("time taken: ", end-start)

if __name__ == "__main__":
    main()
