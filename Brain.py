"""
This file implements a variant of Q-learning for training the neural network
"""
import threading
import random
import time
from MLP import MLP
import pickle as pickle
import numpy as np

#This class provides access to the neural network, and runs a separate thread for training the network
class Brain(threading.Thread):
    def __init__(self, size, alpha, discount, seed):
        random.seed(seed)
        threading.Thread.__init__(self)
        
        self.time = time.time()

        self.quit = False
        self.expectedRewards = {}
        self.targets = {}
        self.experiences = []
        self.error = []

        #If there is previous training data available load it
        try:
            with open("observations/rewards.dat", 'rb') as inFile:
                self.expectedRewards = pickle.load(inFile)

            with open("observations/targets.dat", 'rb') as inFile:
                self.targets = pickle.load(inFile)
            with open("observations/states.dat", 'rb') as inFile:
                self.states = pickle.load(inFile)
                
        except:
            print("rewards targets or states not loaded")
            self.expectedRewards = {}
            self.targets = {}
            self.states = []

     
        self.alpha = alpha
        self.discount = discount
        self.oldScore = 0
        self.oldAction = 0
        self.oldState = [0,]*(100)
        self.oldState = tuple(self.oldState)

        #If no previous data was loaded, store an initial state
        if self.oldState not in self.expectedRewards:
            newTarget = [random.random(),]*5
            self.expectedRewards[self.oldState] = newTarget
            
            #Normalize the rewards so they can be used as targets for the network
            bestReward = float('-inf')
            worstReward = float('inf')
            for x in newTarget:
                if x < worstReward:
                    worstReward = x
                if x > bestReward:
                    bestReward = x

            if bestReward - worstReward == 0:
                pass
            else:
                for x in range(len()):
                    newTarget[x] = (newTarget[x] - worstReward)/(bestReward - worstReward)

            self.targets[self.oldState] = newTarget

        #initialize the neural network
        self.NN = MLP([(size*size), 25, 5], seed)

        #If a previously trained weight matrix exists load it
        name = ""
        for x in self.NN.weights:
            name += (str(np.shape(x)) + ",")
        try:
            with open("observations/"+name, 'rb') as weights:
                self.NN.weights = pickle.load(weights)
        except:
            print("matrix not found")
            pass

    #This method is used to update the set of previously observed sates, and expected rewards for each action
    def updateTable(self, score, state, action):
        #the reward for the pervious state and action
        R = score - self.oldScore

        #now we update the previous state with the currently observed reward
        self.updateTableEntry(R, self.oldState, action, state)

        #Store the passed state action and score, so it can be updated appropriately at the next call
        self.oldScore = score
        self.oldState = state
        self.oldAction = action
    
    #This method performs a forward pass of the neural network to produce an action to take
    def getAction(self, score, state):
        
        #produce a random action 20% of the time
        if random.random() < 0.2:
            rndChoice = int(random.random()*5)
            rndAction = [0,0,0,0,0]
            rndAction[rndChoice] = 1
            return rndAction

        #convert the passed state to the form required by the neural network
        longState = []
        for x in state:
            for y in x:
                longState.append(y)

        #discretize the state
        for x in range(len(longState)):
            if longState[x] > 0.5 and longState[x] <= 1:
                longState[x] = 1
            elif longState[x] <= 0.5 and longState[x] >= 0:
                longState[x] = 0
       
        #get an action from the neural network
        longState = tuple(longState)
        result = self.NN.getAction(longState)
        
        bestScore = float('-inf')
        act = 0
        for x in range(len(result)):
            if result[x] > bestScore:
                bestScore = result[x]
                act = x

        action = [0,]*5
        action[act] = 1

        stateID = longState

        self.experiences.append((self.oldState, act, score - self.oldScore, longState))
        if len(self.experiences) > 100:
            del self.experiences[0]

        #This loop replays recent experiences to update the state/action/reward table
        #over time it helps to propagate future rewards backwards more quickly
        #this is needed since the updateTable method only propagates a future reward backwards one state
        #This means that for the table to properly reflect long term gains the system must experience some state, and take the same action many times
        #The constraints of real time learning make this infeasible
        for x in range(int(len(self.experiences)*0.5)):
            rndExperience = int(random.random()*len(self.experiences))
            experience = self.experiences[rndExperience]
            state = experience[0]
            nextState = experience[3]
            observedReward = experience[2]
            self.updateTableEntry(observedReward, state, experience[1], nextState)
            
                
        #update the state table and return the result to the arbitrator
        self.updateTable(score, longState, act)

        

        if longState not in self.states:
            self.states.append(longState)
        
        print(result)

        return action

    #This method is used to update entries in the state action reward table
    def updateTableEntry(self, observedReward, state, action, nextState):

        #If a new state is passed compute random expected rewards
        if nextState not in self.expectedRewards:
            initialReward = [random.random(),]*5
            self.expectedRewards[nextState] = initialReward
            newTarget = [0,]*5
          
            bestReward = float('-inf')
            worstReward = float('inf')
            for x in initialReward:
                if x < worstReward:
                    worstReward = x
                if x > bestReward:
                    bestReward = x

            if bestReward - worstReward == 0:
                newTarget = initialReward
            else:
                for x in range(len(initialReward)):
                    newTarget[x] = (newTarget[x] - worstReward)/(bestReward - worstReward)

            self.targets[nextState] = newTarget

        #update the expected reward for the appropriate task
        expectedReward = self.expectedRewards[state][action]
        futureReward = float('-inf')
        for y in self.expectedRewards[nextState]:
            if y > futureReward:
                futureReward = y
        self.expectedRewards[state][action] = expectedReward + self.alpha*(observedReward + (self.discount*futureReward) - expectedReward)

        rewards = self.expectedRewards[state]

        newTarget = [0,]*5
        bestReward = float('-inf')
        worstReward = float('inf')
        for x in rewards:
            if x < worstReward:
                worstReward = x
            if x > bestReward:
                bestReward = x
        if bestReward - worstReward == 0:
            pass
        else:
            for x in range(len(rewards)):
                newTarget[x] = (rewards[x] - worstReward)/(bestReward - worstReward)
        
        self.targets[state] = newTarget


    #This method executes in a separtate thread, and periodically trains the neural network
    def run(self):
        error = 1
        while True:
            #Every 30 seconds record the current state table, expected rewards, and weight matrix
            if time.time() - self.time > 30:
                with open("observations/rewards.dat", 'wb') as outFile:
                    pickle.dump(self.expectedRewards, outFile)

                with open("observations/targets.dat", 'wb') as outFile:
                    pickle.dump(self.targets, outFile)

                with open("observations/states.dat", 'wb') as outFile:
                    pickle.dump(self.states, outFile)

                name = ""
                for x in self.NN.weights:
                    name += (str(np.shape(x)) + ",")

                with open("observations/"+name, 'wb') as outFile:
                    pickle.dump(self.NN.weights, outFile)

                self.time = time.time()

            time.sleep(1)
            if self.quit:
                break

            #If the state table is not empty train the network
            if len(self.states) > 0:
                #randomize the expected rewards
                try:
                    sample = len(self.states)
                    
                    inputs = random.sample(self.states, sample)
                    
                    targets = [self.targets[k] for k in inputs]
                except:
                    print("##########################################")
                    continue

                #Train the network for 100 iterations on the random sample
                #note that the random sample contains all 
                if error > 0.05:
                    self.NN.train(inputs, targets, 100, 0.01, 0.1)

                #report the current error
                results = self.NN.getOutputs(inputs, 0.01)

                for x in range(len(targets)):
                    partialSum = 0
                    for y in range(len(targets[0])):
                        partialSum += (targets[x][y] - results[1][x][y])**2
                    error += partialSum

                error = error/len(targets)
                self.error.append(error)
                print("Error: " + str(error))

        print("nn quit")

    def shutdown(self):
        self.quit = True