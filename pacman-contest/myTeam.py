# myTeam.py
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
# from baselineTeam import ReflexCaptureAgent
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import distanceCalculator
import layout
import numpy as np
import random
from util import nearestPoint, manhattanDistance

USE_BELIEF_DISTANCE = True
arguments = {}
predictions = []
predictionsInitialised = []
FORWARD_LOOKING_LOOPS = 3


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='ValueIterationAgent', second='QLearningAgent'):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
    # The following line is an example only; feel free to change it.
    print([eval(first)(firstIndex), eval(second)(secondIndex)])
    # print(DummyAgent.initializationValues(self, gameState))

    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class DummyAgent(CaptureAgent):

    # create coordinate list of all food
    def getFoodPositions(self, gameState):
        foodMatrix = CaptureAgent.getFood(self, gameState)
        self.foodPositions = []
        x = 0
        for row in foodMatrix:
            y = 0
            for j in row:
                if j == True:
                    self.foodPositions.append((x,y))
                y += 1
            x += 1


        return "2"

    # update and return best q-value
    def valueEstimator(self, gameState):
        
        opponents = self.getOpponents(gameState)
        opponent1Pos = gameState.getAgentPosition(opponents[0])
        opponent2Pos = gameState.getAgentPosition(opponents[1])
        
        

        states = self.states_rewards_qvalue
        
        
        

        states = self.states_rewards_qvalue

        for (i, j) in states:

            reward_qvalue = states.get((i, j))
            state_reward = reward_qvalue[0]
            state_q_before = reward_qvalue[1]
            state_q_after = state_q_before

            if states.get((i + 1, j)):
                qsa = states.get((i + 1, j))[0] + (0.9 * states.get((i + 1, j))[1])
                if qsa > state_q_after:
                    state_q_after = qsa

            if states.get((i - 1, j)):
                qsa = states.get((i - 1, j))[0] + (0.9 * states.get((i - 1, j))[1])
                if qsa > state_q_after:
                    state_q_after = qsa

            if states.get((i, j - 1)):
                qsa = states.get((i, j - 1))[0] + (0.9 * states.get((i, j - 1))[1])
                if qsa > state_q_after:
                    state_q_after = qsa

            if states.get((i, j + 1)):
                qsa = states.get((i, j + 1))[0] + (0.9 * states.get((i, j + 1))[1])
                if qsa > state_q_after:
                    state_q_after = qsa

            self.states_rewards_qvalue[(i, j)] = [state_reward, state_q_after]

        if(opponent1Pos) != None and self.capsulePosition != None:
            opponentState = (opponent1Pos[0], opponent1Pos[1])
            self.opponentQDropper(gameState,opponentState)

        
        if(opponent1Pos) != None and self.capsulePosition == None and self.post_capsule_timer < 40:
             reward_qvalue = states[(opponent1Pos[0], opponent1Pos[1])] 
             qvalue = reward_qvalue[1]
             qvalue += 5
             reward_qvalue.pop()
             reward_qvalue.append(qvalue)
             self.states_rewards_qvalue[(opponent1Pos[0], opponent1Pos[1])] = reward_qvalue
        elif (opponent1Pos) != None and self.post_capsule_timer > 40:
            opponentState = (opponent1Pos[0], opponent1Pos[1])
            self.opponentQDropper(gameState,opponentState)
        

        if(opponent2Pos) != None and self.capsulePosition != None:
            opponentState = (opponent2Pos[0], opponent2Pos[1])
            self.opponentQDropper(gameState,opponentState)
        
        
        if(opponent2Pos) != None and self.capsulePosition == None and self.post_capsule_timer < 40:
             reward_qvalue = states[(opponent2Pos[0], opponent2Pos[1])] 
             qvalue = reward_qvalue[1]
             qvalue += 5
             reward_qvalue.pop()
             reward_qvalue.append(qvalue)
             self.states_rewards_qvalue[(opponent2Pos[0], opponent2Pos[1])] = reward_qvalue
        elif (opponent2Pos) != None and self.post_capsule_timer > 40:
            opponentState = (opponent2Pos[0], opponent2Pos[1])
            self.opponentQDropper(gameState,opponentState)

    def opponentQDropper(self,gameState,state):

        south_state_one = (state[0], state[1] - 1)
        if self.states_rewards_qvalue.get(south_state_one):
            
            reward_qvalue = self.states_rewards_qvalue[south_state_one]
            qvalue = reward_qvalue[1]
            qvalue -= 40
            reward_qvalue.pop()
            reward_qvalue.append(qvalue)
            self.states_rewards_qvalue[south_state_one] = reward_qvalue


        south_state_two = (state[0], state[1] - 2)
        if self.states_rewards_qvalue.get(south_state_two):
            reward_qvalue = self.states_rewards_qvalue[south_state_two]
            qvalue = reward_qvalue[1]
            qvalue -= 20
            reward_qvalue.pop()
            reward_qvalue.append(qvalue)
            self.states_rewards_qvalue[south_state_two] = reward_qvalue
        

        east_state_one = (state[0] + 1, state[1])
        if self.states_rewards_qvalue.get(east_state_one):
            reward_qvalue = self.states_rewards_qvalue[east_state_one]
            qvalue = reward_qvalue[1]
            qvalue -= 20
            reward_qvalue.pop()
            reward_qvalue.append(qvalue)
            self.states_rewards_qvalue[east_state_one] = reward_qvalue


        east_state_two = (state[0] + 2, state[1])
        if self.states_rewards_qvalue.get(east_state_two):
            reward_qvalue = self.states_rewards_qvalue[east_state_two]
            qvalue = reward_qvalue[1]
            qvalue -= 10
            reward_qvalue.pop()
            reward_qvalue.append(qvalue)
            self.states_rewards_qvalue[east_state_two] = reward_qvalue

        west_state_one = (state[0] - 1, state[1])
        if self.states_rewards_qvalue.get(west_state_one):
            reward_qvalue = self.states_rewards_qvalue[west_state_one]
            qvalue = reward_qvalue[1]
            qvalue -= 20
            reward_qvalue.pop()
            reward_qvalue.append(qvalue)
            self.states_rewards_qvalue[west_state_one] = reward_qvalue

        west_state_two = (state[0] - 2, state[1])
        if self.states_rewards_qvalue.get(west_state_two):
            reward_qvalue = self.states_rewards_qvalue[west_state_two]
            qvalue = reward_qvalue[1]
            qvalue -= 10
            reward_qvalue.pop()
            reward_qvalue.append(qvalue)
            self.states_rewards_qvalue[west_state_two] = reward_qvalue

        north_state_one = (state[0], state[1] + 1)
        if self.states_rewards_qvalue.get(north_state_one):
            reward_qvalue = self.states_rewards_qvalue[north_state_one]
            qvalue = reward_qvalue[1]
            qvalue -= 10
            reward_qvalue.pop()
            reward_qvalue.append(qvalue)
            self.states_rewards_qvalue[north_state_one] = reward_qvalue

        north_state_two = (state[0], state[1] + 2)
        if self.states_rewards_qvalue.get(north_state_two):
            reward_qvalue = self.states_rewards_qvalue[north_state_two]
            qvalue = reward_qvalue[1]
            qvalue -= 20
            reward_qvalue.pop()
            reward_qvalue.append(qvalue)
            self.states_rewards_qvalue[north_state_two] = reward_qvalue

    # create a dictionary containing all coordinates of possible moves
    def developStatesDict(self, gameState):

        maxY = gameState.data.layout.height - 1
        print(gameState.data.layout.layoutText)
        for y in range(gameState.data.layout.height):
            for x in range(gameState.data.layout.width):
                layoutChar = gameState.data.layout.layoutText[maxY - y][x]
                gameState.data.layout.processLayoutChar(x, y, layoutChar)
                if layoutChar == ' ':
                    self.states_rewards_qvalue[(x, y)] = [0, 0]
                elif layoutChar == '.':
                    self.states_rewards_qvalue[(x, y)] = [0, 0]
                elif layoutChar == 'o':
                    self.states_rewards_qvalue[(x, y)] = [0, 0]
                elif layoutChar in ('1','2','3','4'):
                    self.states_rewards_qvalue[(x, y)] = [0, 0]

        #self.states_rewards_qvalue[(30, 14)] = [0, 0]
        #self.states_rewards_qvalue[(30, 13)] = [0, 0]


        capsulePositionList = CaptureAgent.getCapsules(self, gameState)
        # here abdul!
        if capsulePositionList != []:
            self.states_rewards_qvalue[capsulePositionList[0]] = [5, 0]
        for i in self.foodPositions:
            self.states_rewards_qvalue[i] = [10,0]





        return self.states_rewards_qvalue
    
    def developStatesDictCapsuleEaten(self, gameState):

        self.states_rewards_qvalue = {}
        maxY = gameState.data.layout.height - 1
        for y in range(gameState.data.layout.height):
            for x in range(gameState.data.layout.width):
                layoutChar = gameState.data.layout.layoutText[maxY - y][x]
                gameState.data.layout.processLayoutChar(x, y, layoutChar)
                if layoutChar == ' ':
                    self.states_rewards_qvalue[(x, y)] = [0, 0]
                elif layoutChar == '.':
                    self.states_rewards_qvalue[(x, y)] = [0, 0]
                elif layoutChar == 'o':
                    self.states_rewards_qvalue[(x, y)] = [0, 0]
                elif layoutChar in ('1','2','3','4'):
                    self.states_rewards_qvalue[(x, y)] = [0, 0]


        for i in self.foodPositions:
            self.states_rewards_qvalue[i] = [5,0]

        self.capsulePosition = None
        return self.states_rewards_qvalue

    def registerInitialState(self, gameState):
        """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

        '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
        # initialise previous_action, and starting state characteristics
        self.previous_action = None
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)
        # self.developStatesDict(gameState)
        # self.blueOrRed(gameState)
        capsulePositionList = CaptureAgent.getCapsules(self, gameState)
        if capsulePositionList != []:
            self.capsulePosition = capsulePositionList[0]

    # initialise values maybe used for Q updates
    def __init__( self, index ):
        CaptureAgent.__init__(self, index)
        self.weights = util.Counter()
        self.episodesSoFar = 0
        self.epsilon = 0.06
        # self.gamma = 0.8
        self.alpha = 0.2

    # gives action with biggest q value
    def decideAction(self, state):
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        possibleActions = None
        bestQValue = -100000
        for action in state.getLegalActions(self.index):
            q_value = self.calcQValue(state, action)
            if q_value > bestQValue:
                possibleActions = [action]
                bestQValue = q_value
            elif q_value == bestQValue:
                possibleActions.append(action)
            if possibleActions == None:
                return Directions.STOP
        return random.choice(possibleActions)
    
    #chooses the action based on features and weights and explores with some probability
    # and changes behaviour of game near end
    def chooseAction(self, gameState):
        self.observationHistory.append(gameState)
        # Pick Action
        legalActions = gameState.getLegalActions(self.index)
        action = None
        if len(legalActions):
            if np.random.uniform(0, 1) > self.epsilon:
                action = self.decideAction(gameState)
            else:
                action = random.choice(legalActions)

        self.previous_action = action

        
        foodLeft = len(self.getFood(gameState).asList())
        # Prioritize going back to start if we have <= 2 pellets left
        if foodLeft <= 2:
            bestDist = 9999
            for a in legalActions:
                successor = self.getSuccessor(gameState, a)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start,pos2)
                if dist < bestDist:
                    action = a
                    bestDist = dist

        return action

    def getSuccessor(self, gameState, action):
        """
    Finds the next successor which is a grid position (location tuple).
    """

        successor = gameState.generateSuccessor(self.index, action)

        # print(successor)
        pos = successor.getAgentState(self.index).getPosition()
        # print(pos)
        if pos != util.nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor


    
    def getFoodPositionsHome(self, gameState):
        foodMatrix = CaptureAgent.getFoodYouAreDefending(self, gameState)
        x = 0
        for row in foodMatrix:
            y = 0
            for j in row:
                if j == True:
                    self.home_food_positions.append((x,y))
                y += 1
            x += 1


    def getWeights(self, gameState, action):
        return {'successorScore': 1.0}

    def nextBestQValue(self, state):
        bestValue = -999999
        noLegalActions = True
        for action in state.getLegalActions(self.index):
            noLegalActions = False
            value = self.calcQValue(state, action)
            if value > bestValue:
                bestValue = value
            if noLegalActions:
                return 0         
        return bestValue

    def calcQValue(self, state, action):
        total = 0
        weights = self.getWeightVals()
        features = self.getFeatures(state, action)
        for feature in features:
            total += features[feature] * weights[feature]
        return total
    """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""   
class ValueIterationAgent(DummyAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    #def displayStatesPolicies(self,gameState):
    states_rewards_qvalue = {}
    capsulePosition = (0,0)
    foodPositions  = []
    ghosts_spotted = None
    post_capsule_timer = 0
    home_food_positions = []
    
    post_capsule_timer = 0
    move_timer = 0

    min_food_to_return = 6
    ghost_buffer = 5
    respective_position = 0
    


    

    def getReturnFoodValue(self,myPos,gameState,state):
        if self.numCarrying >= self.min_food_to_return:
            return self.getMazeDistance(self.start, myPos)
        else:
            return 0

    def withinGhostBuffer(self,myPos,min_ghost):
        if min_ghost > self.ghost_buffer or min_ghost == 0:
            return 0
        else: self.getMazeDistance(self.start,myPos)
 
    def registerInitialState(self, gameState):
        """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

        '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''

        CaptureAgent.registerInitialState(self, gameState)


        '''
    Your initialization code goes here, if you need any.
    '''
        self.start = gameState.getAgentPosition(self.index)
        self.respective_position = self.start[0] - gameState.data.layout.width
        print(self.developStatesDict(gameState))
        # print(self.blueOrRed(gameState))

        self.valueEstimator(gameState)
        self.valueEstimator(gameState)
        self.valueEstimator(gameState)
        self.valueEstimator(gameState)
    

        print("Capsules: ")
        capsulePositionList = CaptureAgent.getCapsules(self, gameState)
        print(capsulePositionList)
        print(capsulePositionList)
        if capsulePositionList != []:
            self.capsulePosition = capsulePositionList[0]



        


        # print(self.blueOrRed(gameState))


        print("!!!!!",self.states_rewards_qvalue.get((30, 14)))




        print(self.states_rewards_qvalue)

        print("Opponents: ")
        print(CaptureAgent.getOpponents(self, gameState))
        print("Actions: ")
        print(CaptureAgent.getAction(self, gameState))
        print("Food: ")
        print(CaptureAgent.getFood(self, gameState))
        
        print("Current Observations: ")
        print(CaptureAgent.getCurrentObservation(self))
        print("Previous Observations: ")
        print(CaptureAgent.getPreviousObservation(self))
        print("Score: ")
        print(CaptureAgent.getScore(self, gameState))
        print("Team: ")
        print(CaptureAgent.getTeam(self, gameState))


    def decreaseQValue(self, index, amount):
        reward_qvalue = self.states_rewards_qvalue.get(index)
        qvalue = reward_qvalue[1]
        qvalue -= amount
        reward_qvalue.pop()
        reward_qvalue.append(qvalue)
        self.states_rewards_qvalue[index] = reward_qvalue

    def offensiveFoodHeuristic(self, state, foodList):
        minDistance = 9999
        for food in foodList:
            if self.getMazeDistance(state,food) < minDistance:
                minDistance = self.getMazeDistance(state,food)
        return minDistance    

    def homeHeurisitic(self,state,gameState):
        self.getFoodPositionsHome(gameState)
        goal_state = self.home_food_positions.pop()
        minDistance = self.getMazeDistance(state,goal_state)
        return minDistance

    def climbTheHillHome(self, gameState, actions, agentPosition):
        hillFIFO = util.Stack()
        for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                hillFIFO.push((pos2,action))
        closed_list = []
        while not hillFIFO.isEmpty():
            (state,action) = hillFIFO.pop()
            if state not in closed_list:
                closed_list.append(state[0])
                if self.homeHeurisitic(state,gameState) < self.homeHeurisitic(agentPosition,gameState):
                    if state in self.foodPositions:
                        index = self.foodPositions.index(state)
                        self.foodPositions.pop(index)
                    return [state,action]
                
        return "Stop"

        
    def enforcedHillClimbing(self, gameState, actions, agentPosition):
        hillFIFO = util.Stack()
        for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                hillFIFO.push((pos2,action))
        closed_list = []
        while not hillFIFO.isEmpty():
            (state,action) = hillFIFO.pop()
            if state not in closed_list:
                closed_list.append(state[0])
                if self.offensiveFoodHeuristic(state,self.foodPositions) < self.offensiveFoodHeuristic(agentPosition,self.foodPositions):
                    if state in self.foodPositions:
                        index = self.foodPositions.index(state)
                        self.foodPositions.pop(index)
                    return [state,action]
                
        return "Stop"
                
    def goHome(self,gameState):
        TIME_RUN_OUT = 900.0
        numCarrying = gameState.getAgentState(self.index).numCarrying
        winning_margin = self.getScore(gameState)


        return (gameState.data.timeleft < TIME_RUN_OUT and winning_margin <= 0 and numCarrying > 0 and numCarrying >= abs(winning_margin))

    def chooseAction(self, gameState):
        """
    Picks among actions randomly
    """
        # actions = gameState.getLegalActions(self.index)

        '''
    You should change this in your own agent.
    '''
        
        # return random.choice(actions)

        """
    Picks among the actions with the highest Q(s,a).
    """
        #print(CaptureAgent.getCurrentObservation(self))
        
        #if self.goHome(gameState):
        

        capsulePositionList = CaptureAgent.getCapsules(self, gameState)
        if capsulePositionList != []:
            self.capsulePosition = capsulePositionList[0]
        else:
            self.capsulePosition = None
        self.getFoodPositions(gameState)
        self.valueEstimator(gameState)
        self.valueEstimator(gameState)
        actions = list(gameState.getLegalActions(self.index))
        agentPosition = gameState.getAgentPosition(self.index)
        self.move_timer += 1
        if self.move_timer % 25 == 0:
            self.developStatesDict(gameState)

        if self.goHome(gameState):
            print("Going home")
            state_action = self.climbTheHillHome(gameState,actions,agentPosition)
            action = state_action[1]
            return action
        
        agent_nearby = []


        distance_states = []
        south_state_three = (agentPosition[0], agentPosition[1] - 3)
        distance_states.append(south_state_three)
    
        east_state_three = (agentPosition[0] + 3, agentPosition[1])
        distance_states.append(east_state_three)

        west_state_three = (agentPosition[0] - 3, agentPosition[1])
        distance_states.append(west_state_three)

        north_state_three = (agentPosition[0], agentPosition[1] + 4)
        distance_states.append(north_state_three)



        if self.capsulePosition == None:
            self.post_capsule_timer += 1

        if agentPosition in distance_states:
            self.developStatesDict(gameState)
            self.valueEstimator(gameState)
            self.valueEstimator(gameState)
        else:
            self.valueEstimator(gameState)


        print("Agent Position:")
        print(agentPosition)


        

        

        south_state = (agentPosition[0], agentPosition[1] - 1)
        agent_nearby.append(south_state)

        east_state = (agentPosition[0] + 1, agentPosition[1])
        agent_nearby.append(east_state)

        west_state = (agentPosition[0] - 1, agentPosition[1])
        agent_nearby.append(west_state)

        north_state = (agentPosition[0], agentPosition[1] + 1)
        agent_nearby.append(north_state)

        south_state_two = (agentPosition[0], agentPosition[1] - 2)
        agent_nearby.append(south_state_two)

        east_state_two = (agentPosition[0] + 2, agentPosition[1])
        agent_nearby.append(east_state_two)

        west_state_two = (agentPosition[0] - 2, agentPosition[1])
        agent_nearby.append(west_state_two)

        north_state_two = (agentPosition[0], agentPosition[1] + 2)
        agent_nearby.append(north_state_two)

        



        print("Opponents:")
        opponents = self.getOpponents(gameState)
        opponent1Pos = gameState.getAgentPosition(opponents[0])
        opponent2Pos = gameState.getAgentPosition(opponents[1])

        print([opponent1Pos,opponent2Pos])

        if self.offensiveFoodHeuristic(agentPosition,self.foodPositions) > (0.75 * gameState.data.layout.width):
            state_action = self.enforcedHillClimbing(gameState,actions,agentPosition)
            if (opponent1Pos or opponent2Pos) not in agent_nearby:
                print("Climbing Hill")
                action = state_action[1]
                return action

        # print(gameState.getWalls())
        # print(gameState.data.layout.layoutText)

        argmax = {}


        flag = False


        if south_state == self.capsulePosition or south_state in self.foodPositions:
            flag = True
        if east_state == self.capsulePosition or east_state in self.foodPositions:
            flag = True
        if west_state == self.capsulePosition or west_state in self.foodPositions:
            flag = True
        if north_state == self.capsulePosition or north_state in self.foodPositions:
            flag = True




        if flag:

            if south_state == self.capsulePosition:
                self.developStatesDictCapsuleEaten(gameState)
                self.post_capsule_timer += 1
                return "South"
            elif south_state in self.foodPositions:
                print((south_state))
                index = self.foodPositions.index(south_state)
                print(index)
                self.foodPositions.pop(index)
                """
                self.decreaseQValue(south_state,150)
                self.decreaseQValue(agentPosition,150)
                self.states_rewards_qvalue[south_state] = [0,0]
                self.states_rewards_qvalue[agentPosition] = [0,0]
                """
                self.developStatesDict(gameState)
                # self.gets
                return "South"


            if east_state == self.capsulePosition:
                self.developStatesDictCapsuleEaten(gameState)
                self.post_capsule_timer += 1
                return "East"
            elif east_state in self.foodPositions:
                print((east_state))
                index = self.foodPositions.index(east_state)
                print(index)
                self.foodPositions.pop(index)
                """
                self.decreaseQValue(east_state,150)
                self.decreaseQValue(agentPosition,150)
                self.states_rewards_qvalue[east_state] = [0,0]
                self.states_rewards_qvalue[agentPosition] = [0,0]
                """

                self.developStatesDict(gameState)
                return "East"


            if west_state == self.capsulePosition:
                self.developStatesDictCapsuleEaten(gameState)
                return "West"
            elif west_state in self.foodPositions:
                print((west_state))
                index = self.foodPositions.index(west_state)
                print(index)
                self.foodPositions.pop(index)
                """
                self.decreaseQValue(west_state,150)
                self.decreaseQValue(agentPosition,150)
                self.states_rewards_qvalue[west_state] = [0,0]
                self.states_rewards_qvalue[agentPosition] = [0,0]
                """
                self.developStatesDict(gameState)
                return "West"


            if north_state == self.capsulePosition:
                self.developStatesDictCapsuleEaten(gameState)
                return "North"
            elif north_state in self.foodPositions:
                print((north_state))
                index = self.foodPositions.index(north_state)
                print(index)
                self.foodPositions.pop(index)
                """
                self.decreaseQValue(north_state,150)
                self.decreaseQValue(agentPosition,150)
                self.states_rewards_qvalue[north_state] = [0,0]
                self.states_rewards_qvalue[agentPosition] = [0,0]
                """
                self.developStatesDict(gameState)
                return "North"


        print("Actions are")
        self.valueEstimator(gameState)
        for action in actions:
            print(action)
            successor = self.getSuccessor(gameState, action)
            pos2 = successor.getAgentPosition(self.index)
            print(pos2)

        dictOfR = {}
        for action in actions:

            if action == "South":
                reward = self.states_rewards_qvalue.get(south_state)
                if reward:
                    dictOfR["South"] = reward
                    argmax["South"] = reward[1]

            if action == "East":
                reward = self.states_rewards_qvalue.get(east_state)
                if reward:
                    dictOfR["East"] = reward
                    argmax["East"] = reward[1]

            if action == "West":
                reward = self.states_rewards_qvalue.get(west_state)
                if reward:
                    dictOfR["West"] = reward
                    argmax["West"] = reward[1]

            if action == "North":
                reward = self.states_rewards_qvalue.get(north_state)
                if reward:
                    dictOfR["North"] = reward
                    argmax["North"] = reward[1]

        print(dictOfR)
        maximum = max(argmax, key=argmax.get)
        print(maximum)
        #print(maximum, argmax[maximum])

        if maximum == opponent1Pos or maximum == opponent2Pos:
            self.developStatesDict(gameState)
        return maximum
    



"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""

# Agent that deals with features, weights and behaviours
class QLearningAgent(DummyAgent):
    def registerInitialState(self, gameState):
        DummyAgent.registerInitialState(self, gameState)
        self.time_to_defend = 0.0
        self.recent_returned_food = 0.0
        self.getMovablePositions(gameState)
        self.Y_axis_tendency = gameState.data.layout.height / 2

    def __init__( self, index ):
        DummyAgent.__init__(self, index)
        self.weights = util.Counter()
        self.capsule_care_range = 5
        self.weights['food_distance'] = -1
        self.weights['ghost_distance'] = 5
        self.weights['stop'] = -1000
        self.weights['legal_actions'] = 100
        self.weights['capsule_value'] = 20
        # self.weights['defence_capsule'] = -.5
        self.weights['attacking'] = -5
        self.weights['enemy_pacman_dist'] = -100
        self.weights['get_safe'] = -1
        self.weights['y_distance_to_food'] = -1
        # self.weights['defence_centre'] = -25
        self.min_food_to_return = 6
        self.weights['chase_scared_enemy'] = -100
        self.weights['successor_score'] = 100
        self.ghost_buffer = 5
        self.legal_action_map = {}
        self.init_legal_pos = False

    # returns list of legal positions (i.e. positions that arent walls)
    def getMovablePositions(self, gameState):
        if not self.init_legal_pos:
            self.legal_pos = []
            walls = gameState.getWalls()
            for i in range(walls.width):
                for j in range(walls.height):
                    if not walls[i][j]:
                        self.legal_pos.append((i, j))
            self.init_legal_pos = True
        return self.legal_pos

    # dictionary with legal actions as values for positions 
    def getLegalActions(self, gameState):
        currentPos = gameState.getAgentState(self.index).getPosition()
        if currentPos not in self.legal_action_map:
            self.legal_action_map[currentPos] = gameState.getLegalActions(self.index)
        return self.legal_action_map[currentPos]
# ASH does above this line!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    def getFeatures(self, gameState, action):
        #Get all features
        MAX_TIME_IN_DEFENCE = 100.0
        MULT = 2
        VISION = 5

        features = util.Counter()
        self.predictOpponents(gameState)
        successor = self.getSuccessor(gameState, action)
        state = successor.getAgentState(self.index)
        pos = state.getPosition()
        print("pos = ",pos)
        food_list = self.getFood(successor).asList()    
        features['successor_score'] = -len(food_list)

        if len(food_list) > 0:
            dist = min([self.getMazeDistance(pos, food) for food in food_list])
            features['food_distance'] = dist

        # if len(gameState.getCapsules()) > 0:
        #     features['defence_capsule'] = min([self.getMazeDistance(pos,c) for c in gameState.getCapsules()])

        if self.red:
            if pos[0] > gameState.data.layout.width/2:
                features['attack'] = manhattanDistance(pos, (gameState.data.layout.width/2,pos[1]))
        else:
            if pos[0] < gameState.data.layout.width/2:
                features['attack'] = manhattanDistance(pos, (gameState.data.layout.width/2,pos[1]))
            
        #Obtaining all enemies:
        allEnemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemy_pacman = [a for a in allEnemies if a.isPacman and a.getPosition() != None]
        defence_ghost = [a for a in allEnemies if not a.isPacman and a.getPosition() != None and not a.scaredTimer > 0]
        scared_ghosts = [a for a in allEnemies if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]

        features['chase_scared_enemy'] = self.attackWeight(pos, enemy_pacman)
        features['capsule_value'] = self.getCapsuleValue(pos, successor, scared_ghosts)

        #Calculate enemy pacman distance feature if enemy pacman is on out grid
        pac_dist = 1000
        # print("------------------------")
        if len(enemy_pacman) == 0:
            features['enemy_pacman_dist'] = 0
        else:
            for pacman in enemy_pacman:
                # print(pacman.isPacman)
                if pacman.getPosition() != None:
                    pac_dist = min(pac_dist, self.getMazeDistance(pos, pacman.getPosition()))
                    features['enemy_pacman_dist'] = pac_dist
                else:
                    features['enemy_pacman_dist'] = MULT * VISION 
            

        #calculate min ghost diatance feature
        distances = []
        for index in self.getOpponents(successor):
            ghost = successor.getAgentState(index)
            if ghost in defence_ghost:
                distances.append(self.getMazeDistance(pos, ghost.getPosition()))
        if len(distances) > 0:
            min_distance = min(distances)
            features['ghost_distance'] = min_distance
    
        #Feature to decide to chase the scared enemy ghost 
        if state.numReturned != self.recent_returned_food:
            self.time_to_defend = MAX_TIME_IN_DEFENCE
            self.recent_returned_food = state.numReturned
        if self.time_to_defend > 0:
            self.time_to_defend -= 1
            features['chase_scared_enemy'] *= 100
        if len(self.getFoodYouAreDefending(successor).asList()) <= 2:
            features['chase_scared_enemy'] *= 100

        if action == Directions.STOP: 
            features['stop'] = 1
    
        # 3 loops better than 5. 1 was too dangerous and 5 too slow
        features['legal_actions'] = self.numLoopedLegalActions(gameState, FORWARD_LOOKING_LOOPS)

        #Feature to get to safe position
        features['get_safe'] = self.getReturnFoodValue(pos, gameState, state)

        # pt = (int(gameState.data.layout.width/4),int(gameState.data.layout.height/2))
        # pt_2 = (int(0.75*gameState.data.layout.width),int(gameState.data.layout.height/2))

        # half_red = nearestPoint(pt)
        # half_blue = nearestPoint(pt_2)
        
        # if self.red:
        #     features['defence_centre'] = manhattanDistance(pos, half_red)
        # else:
        #     features['defence_centre'] = manhattanDistance(pos, half_blue)
        #     print("defence center = ", features['defence_centre'])

        features['get_safe'] += self.withinGhostBuffer(pos, features['ghost_distance'])
        
        if self.goHome(gameState):
            features['get_safe'] = self.getMazeDistance(self.start, pos) * 10000

        return features
    
    def getWeightVals(self):
        return self.weights

    def goHome(self, gameState):
        #Go home to safety if the conditios are meat
        TIME_RUN_OUT = 80.0
        numCarrying = gameState.getAgentState(self.index).numCarrying
        winning_margin = self.getScore(gameState)
        return (gameState.data.timeleft < TIME_RUN_OUT and winning_margin <= 0 and numCarrying > 0 and numCarrying >= abs(winning_margin))

    def getCapsuleValue(self, myPos, successor, scared_ghosts):
        #Return maximum capsule value
        min_dist = 0
        capsules = self.getCapsules(successor)

        if len(capsules) > 0 and len(scared_ghosts) == 0:
            distances = [self.getMazeDistance(myPos, capsule) for capsule in capsules]
            min_dist = min(distances)
        return max(self.capsule_care_range - min_dist, 0)

    def getReturnFoodValue(self, myPos, gameState, state):
        #Return food value to determine whether to get safe
        if state.numCarrying >= self.min_food_to_return:
            return self.getMazeDistance(self.start, myPos)
        else:
            return 0

    def withinGhostBuffer(self, myPos, min_ghost):
        #calculate ghost buffer distance to determine whether to get safe.
        if min_ghost > self.ghost_buffer or min_ghost == 0:
            return 0
        else:
            return self.getMazeDistance(self.start, myPos) * 1000

    def attackWeight(self, myPos, enemy_pacman):
        #Calculate attack weight to odetermine whether to attack scared ghosts
        if len(enemy_pacman) > 0:
            dists = [self.getMazeDistance(myPos, enemy.getPosition()) for enemy in enemy_pacman]
            if len(dists) > 0:
                smallest_dist = min(dists)
                return smallest_dist
        return 0

    def numLoopedLegalActions(self, gameState, numLoops):
        legalActions = self.getLegalActions(gameState)
        numActions = len(legalActions)
        for legalAction in legalActions:
            if numLoops > 0:
                newState = self.getSuccessor(gameState, legalAction)
                numActions += self.numLoopedLegalActions(newState, numLoops - 1)
        return numActions



    def initialisePredictions(self, gameState):
        #Initialize possible predictions
        predictions.extend([None for x in range(len(self.getOpponents(gameState)) + len(self.getTeam(gameState)))])
        for opp in self.getOpponents(gameState):
            self.initialisePrediction(opp, gameState)
        predictionsInitialised.append('done')

    def initialisePrediction(self, opponentIndex, gameState):
        #For one possible prediction
        predict = util.Counter()
        for pos in self.getMovablePositions(gameState):
           predict[pos] = 1.0
        predict.normalize()
        predictions[opponentIndex] = predict

    def predictOpponents(self, gameState):
        #return predicted opponents if possible predictions are initialised. 
        if len(predictionsInitialised):
            for opp in self.getOpponents(gameState):
                self.predictOneOpponent(gameState, opp)
        else: 
            self.initialisePredictions(gameState)

    def predictOneOpponent(self, gameState, opponentIndex):
        #return predicted opponent
        MIN_PROB = .002
        pacman_pos = gameState.getAgentPosition(self.index)
        poss_positions = util.Counter()
        approx_pos = gameState.getAgentPosition(opponentIndex)
        if approx_pos != None:
            poss_positions[approx_pos] = 1
            predictions[opponentIndex] = poss_positions
            return
        approx_dist = gameState.getAgentDistances()[opponentIndex]
        for pos in self.getMovablePositions(gameState):
            # For each legal ghost position, calculate distance to that ghost
            dist = util.manhattanDistance(pos, pacman_pos)
            prob = gameState.getDistanceProb(dist, approx_dist) 
            if prob > 0:
                oldProb = predictions[opponentIndex][pos]
                poss_positions[pos] = (oldProb + MIN_PROB) * prob
            else:
                poss_positions[pos] = 0
        poss_positions.normalize()
        predictions[opponentIndex] = poss_positions











"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
class NotWorkingQLearningAgent(DummyAgent):
    epsilon = 0.9
    total_episodes = 10000
    max_steps = 100
    alpha = 0.85
    gamma = 0.95

    def registerInitialState(self, gameState):

        CaptureAgent.registerInitialState(self, gameState)
        self.q_values = util.Counter()
        self.start = gameState.getAgentPosition(self.index)

    def calcQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.q_values[(state, action)]

    def nextBestQValue(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        q_vals = []
        for action in self.getLegalActions(state):
            q_vals.append(self.calcQValue(state, action))
        if len(self.getLegalActions(state)) == 0:
            return 0.0
        else:
            return max(q_vals)

    def computeActionFromQValues(self, state, gameState):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        max_action = None
        max_q_val = 0
        for action in gameState.getLegalActions(self.index):
            q_val = self.calcQValue(state, action)
            if q_val > max_q_val or max_action is None:
                max_q_val = q_val
                max_action = action
        return max_action

    def getAction(self, gameState):
        state = gameState.getAgentState(self.index)
        legalActions = gameState.getLegalActions(self.index)
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state, gameState)
        
        successor = self.getSuccessor(gameState, action)
        reward = float(self.getReward(gameState, successor))
        first_part = (1 - self.alpha) * self.calcQValue(state, action)
        if len(legalActions) == 0:
            intermediate = reward
        else:
            intermediate = reward + (self.gamma * max([self.calcQValue(successor, next_action) for next_action in legalActions]))
        second_part = self.alpha * intermediate
        self.q_values[(state, action)] = first_part + second_part
        return action
    
    def getReward(self, gameState, successor):
        reward = 0.0
        enemy_indices = self.getOpponents(gameState)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        pos = gameState.getAgentPosition(self.index)
        # invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        invaders = [a for a in enemies if a.isPacman]
        # print(successor.getAgentPosition(1))
        if len(invaders) > 0:
            for a in enemy_indices:
                if gameState.getAgentState(a).isPacman:
                    print(successor.getAgentPosition(a))
                    if successor.getAgentPosition(a) == None:
                        reward = float(-self.getMazeDistance(pos, ((float)(gameState.data.layout.width/2), (float)(gameState.data.layout.height/2))))
                    else:
                        print("agent pos = ",successor.getAgentPosition(a))
                        print("pos = ", pos)
                        dists = []
                        for a in enemy_indices:
                            enemy_pos = successor.getAgentPosition(a)
                            dists.append(self.getMazeDistance(pos,enemy_pos))
                        # dists = [self.getMazeDistance(pos, successor.getAgentPosition(a)) for a in enemy_indices]
                        reward = float(-min(dists))
        else:
            reward = float(-self.getMazeDistance(pos, ((float)(gameState.data.layout.width/2), (float)(gameState.data.layout.height/2))))
        return reward

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        first_part = (1 - self.alpha) * self.calcQValue(state, action)
        if len(self.getLegalActions(nextState)) == 0:
            sample = reward
        else:
            sample = reward + (self.gamma * max([self.calcQValue(nextState, next_action) for next_action in self.getLegalActions(nextState)]))
        second_part = self.alpha * sample
        self.q_values[(state, action)] = first_part + second_part

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.nextBestQValue(state)















'''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''




class SARSAAgentDefensive(DummyAgent):
    epsilon = 0.9
    total_episodes = 10000
    max_steps = 100
    alpha = 0.85
    gamma = 0.95
    Q = None

  
    #Initializing the Q-matrix 
    
    def registerInitialState(self, gameState):
      self.Q = self.developDefensiveDict(gameState)

      """
      This method handles the initial setup of the
      agent to populate useful fields (such as what team
      we're on).

      A distanceCalculator instance caches the maze distances
      between each pair of positions, so your agents can use:
      self.distancer.getDistance(p1, p2)

      IMPORTANT: This method may run for at most 15 seconds.
      """

      '''
      Make sure you do not delete the following line. If you would like to
      use Manhattan distances instead of maze distances in order to save
      on initialization time, please take a look at
      CaptureAgent.registerInitialState in captureAgents.py.
      '''

      CaptureAgent.registerInitialState(self, gameState)
        # print(gameState.data.layout.height, " height!")

      '''
      Your initialization code goes here, if you need any.
      '''
      self.start = gameState.getAgentPosition(self.index)
    
    
    def chooseAction(self, gameState):
      # successor = self.getSuccessor(gameState, action)
      pos = gameState.getAgentPosition(self.index)
      print("pos = ",pos)
      print("1")
      q_val = self.Q[pos[0],pos[1]]
      print("2")
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      print("3")
      # invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
      invaders = [a for a in enemies if a.isPacman]
      print("4")
      if len(invaders) > 0:
        print("5")
        print("a.getPosition = ", invaders[0].getPosition())
        dists = [self.getMazeDistance(pos, ((float)(a.getPosition()[0]), (float)(a.getPosition()[1]))) for a in invaders]
        print("6")
        reward = -min(dists)
        print("7")
      else:
        print("8")
        reward = -self.getMazeDistance(pos, ((float)(gameState.data.layout.width/2), (float)(gameState.data.layout.height/2)))
        print("9")
          


      # print(pos, " pos_orig")
      actions = gameState.getLegalActions(self.index)
      print("10")
      action = None
      q_max = -100000000000000
      print("11")
      new_pos = pos
      print("12")
      if np.random.uniform(0, 1) < self.epsilon:
        print("13")
      # if 0==1:
        for a in actions:
          print("14")
          # print("acgtion = ",actions)
          if a == "North":
            print("15")
            posN = (pos[0], pos[1]+1)
            print("16")
            print("q = ",self.Q[posN])
            if self.Q[posN] >= q_max:
              print("17")
              # print("action = ",action, " QposN = ",self.Q[posN], " qmax = ", q_max)
              q_max = self.Q[posN]
              print("18")
              action = a
              print("19")
              new_pos = posN
              print("20")
              print(action, new_pos)
            
          if a == 'South':
            posS = (pos[0], pos[1]-1)
            print("21")
            if self.Q[posS] >= q_max:
              q_max = self.Q[posS]
              action = a
              print("22")
              new_pos = posS
              print(action, new_pos)
          if a == 'East':
            posE = (pos[0]+1, pos[1])
            print("23")
            if self.Q[posE] >= q_max:
              q_max = self.Q[posE]
              action = a
              print("24")
              new_pos = posE
              print(action, new_pos)
          if a == 'West':
            posW = (pos[0]-1, pos[1])
            if self.Q[posW] >= q_max:
              print("25")
              q_max = self.Q[posW]
              action = a
              print("26")
              new_pos = posW
              print(action, new_pos)
          self.Q[pos] = (float)(self.Q[pos]) + (float)(self.alpha)*(reward + (float)(self.gamma)*(float)(self.Q[new_pos])-(float)(self.Q[pos]))
          #print(action)
          print("27")
          return action
        else:
          print("28")
          return random.choice(actions)



# class DummyAgentOffensive(DummyAgent):
#     """
#   A reflex agent that seeks food. This is an agent
#   we give you to get an idea of what an offensive agent might look like,
#   but it is by no means the best or only way to build an offensive agent.
#   """


#     def chooseAction(self, gameState):
#       actions = gameState.getLegalActions(self.index)
#       return random.choice(actions)


#       """
        
        
#         # You can profile your evaluation time by uncommenting these lines
#         # start = time.time()
#         values = [self.evaluate(gameState, a) for a in actions]
#         # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

#         maxValue = max(values)
#         bestActions = [a for a, v in zip(actions, values) if v == maxValue]

#         foodLeft = len(self.getFood(gameState).asList())

#         opponents = self.getOpponents(gameState)
#         # print(gameState.getAgentPosition(opponents[0]))
#         # distanceCalculator.manhattanDistance(self.index,opponents)
#         # print(layout.Layout.processLayoutText(self))
#         # print(CaptureAgent.get)

#         if foodLeft <= 2:
#             bestDist = 9999
#             for action in actions:
#                 successor = self.getSuccessor(gameState, action)
#                 pos2 = successor.getAgentPosition(self.index)
#                 dist = self.getMazeDistance(self.start, pos2)

#                 # print(oponentDistance)

#                 if dist < bestDist:
#                     bestAction = action
#                     bestDist = dist
#                     # print("flevaefv________________")

#             # print(bestAction)
#             # print(oponentDistance)
#             return bestAction
#         # oponentDistance = self.getMazeDistance(pos2, opponents)
#         # print(bestAction)
#         # print(DummyAgentOffensive.getFeatures(self,gameState,actions))
#         return random.choice(bestActions)
#         """
#     def getFeatures(self, gameState, action, null=None, opponentDistance=None):
#         print("error")
#         features = util.Counter()

#         # successor = self.getSuccessor(gameState, action)
#         # # print(successor)
#         # opponents = self.getOpponents(gameState)
#         # # print(gameState.getAgentPosition(opponents[0]))
#         # opponentPos = gameState.getAgentPosition(opponents[0])
#         # foodList = self.getFood(successor).asList()
#         # features['successorScore'] = -len(foodList)  # self.getScore(successor)

#         # # Compute distance to the nearest food
#         # opponentDistance
#         # if len(foodList) > 0:  # This should always be True,  but better safe than sorry
#         #     myPos = successor.getAgentState(self.index).getPosition()
#         #     minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
#         #     # opponentDistance = self.getMazeDistance(myPos, opponentPos[0])
#         #     # opponentPos = self.getOpponents(gameState)
#         #     # print(opponentPos)
#         #     if (opponentPos != null):
#         #         opponentDistance = self.getMazeDistance(myPos, opponentPos)
#         #         # print(opponentDistance)
#         #         features['distanceToOpponent'] = opponentDistance

#         #     else:
#         #         features['distanceToOpponent'] = -1

#         #     # opponentDistanceM = distanceCalculator.manhattanDistance(myPos,opponentPos)
#         #     features['distanceToFood'] = minDistance

#         #     # print(opponentDistance)
#         #     # print(features)
#         #     # print(myPos)
#         #     # print(opponentPos)
#         #     # print(opponentDistance)
#         #     # print(foodList)
#         return features

#     def getWeights(self, gameState, action):
#         return {'successorScore': 100, 'distanceToOpponent': -1, 'distanceToFood': -1}


# class DummyAgentDefensive(DummyAgent):
#     """
#   A reflex agent that keeps its side Pacman-free. Again,
#   this is to give you an idea of what a defensive agent
#   could be like.  It is not the best or only way to make
#   such an agent.
#   """

#     def getFeatures(self, gameState, action):
#         features = util.Counter()
#         successor = self.getSuccessor(gameState, action)
#         # print(successor)

#         myState = successor.getAgentState(self.index)
#         myPos = myState.getPosition()

#         # Computes whether we're on defense (1) or offense (0)
#         features['onDefense'] = 1
#         if myState.isPacman: features['onDefense'] = 0

#         # Computes distance to invaders we can see
#         enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
#         invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
#         features['numInvaders'] = len(invaders)
#         if len(invaders) > 0:
#             dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
#             features['invaderDistance'] = min(dists)

#         if action == Directions.STOP: features['stop'] = 1
#         rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
#         if action == rev: features['reverse'] = 1

#         return features

#     def getWeights(self, gameState, action):
#         return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

# class UpperAgent(QLearningAgent):

#     def registerInitialState(self, gameState):
#         super().registerInitialState(gameState)
#         self.Y_axis_tendency = gameState.data.layout.height

# class LowerAgent(QLearningAgent):
#     def registerInitialState(self, gameState):
#         super().registerInitialState(gameState)
#         self.Y_axis_tendency = 0.0
