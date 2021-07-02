"""
    This script implements a feed forward neural network using rProp as the learning algorithm.
"""

import numpy as np
from threading import Lock
import random

class MLP():
    def __init__(self, structure, seed):

        np.random.seed(seed)
        random.seed(seed)

        self.lock = Lock()

        self.structure = structure
        self.beta = 0.1
        self.weights = []
        self.weightsForUse = []

        for x in range(len(structure)-1):
            self.weights.append(np.random.rand(structure[x]+1, structure[x+1])-0.5)
            self.weightsForUse.append(np.random.rand(structure[x]+1, structure[x+1])-0.5)

    #This method is used to train the network
    #inputs must be a list of input values
    #targets must be a list of target values, ordered to correspond with the inputs
    #numEpochs indicates the number of runs to perform with the input data
    #beta is a parameter used to control for overflow in the logistic function
    #dropout is a parameter which specifies the size of a randomly chosen portion of the network to not be trained
    def train(self, inputs, targets, numEpochs, beta, dropout):
        
        alpha = 0.01
       
        
        alphas = []
        
        #construct arrays containing the initial learning rates and momentums
        for x in range(len(self.weights)):
            alphas.append(np.full(np.shape(self.weights[x]), 0.01))
            

        oldGrads = [] 

        #an array to hold the change in weights
        deltaWs = []
        for x in range(len(self.weights)):
            deltaWs.append(np.zeros((np.shape(self.weights[x]))))

        #modify inputs to align properly with the bias nodes
        inputsWithBias = np.concatenate((inputs, np.ones((len(inputs), 1))), axis=1)

        #for each epoch train the network
        for x in range(numEpochs):
            
            #an array for temporary weights for implementing dropout
            tmpWeights = []
            for y in range(len(self.weights)):
                tmpWeights.append(np.copy(self.weights[y]))
                rand = np.random.rand(len(self.weights[y]), len(self.weights[y][0]))
                #if some random value is less than the given dropout rate replace the weight with a zero
                tmpWeights[y][rand < dropout] = 0

            #perform the forward pass
            outputs = self.getOutputs(inputs, beta)

            ErrsWRTweights = []

            #Calculate errors at the output layer
            #ie the DeltaOs
            ErrOutput = (outputs[-1] - targets)*outputs[-1]*(1.0 - outputs[-1])
            ErrsWRTweights.insert(0, ErrOutput)

           
            #get the portion of error for each node based on the values of the weights and the error at each output node
            ErrPortion = np.dot(ErrsWRTweights[0], np.transpose(tmpWeights[1]))

         
            #calculate deltahs
            DerWRTNode = beta*outputs[0]*(1.0 - outputs[0])*ErrPortion
            ErrsWRTweights.insert(0, DerWRTNode)

          
            #calculate the weight updates
            #compute the sum of the gradients for all of the given inputs at each node
            #use this sum to compute the weight update
            gradSums = []
            gradSums.append(np.dot(np.transpose(inputsWithBias),DerWRTNode[:,:-1]))
            
            gradSums.append(np.dot(np.transpose(outputs[0]),ErrsWRTweights[1]))
            
            #If there are gradients from a previous run
            #ie this is not the first run
            if len(oldGrads) > 0: 
                #compute the change in sign of gradient for each weight
                for y in range(len(self.weights)):
                    sign = gradSums[y]*oldGrads[y]
                
                    #where the sign has not changed increase the learning rate
                    alphas[y][sign>0] = alphas[y][sign>0]*1.1
         
                    #where the sign has changed decrease the learning rate
                    alphas[y][sign < 0]*= 0.9
                                       
            oldGrads = gradSums

            #compute the new weight
            deltaWs[0] = alphas[0]*(gradSums[0]) 
            deltaWs[1] = alphas[1]*(gradSums[1]) 

            #finally we update the weights
            for y in range(len(self.weights)):
                self.weights[y] -= deltaWs[y]

        #After training completes record a copy of the weights for use by the robot
        #This ensures that the robot does not attempt to generate an action while weights are being updated
        self.lock.acquire()
        newWeights = []
        for x in self.weights:
            newWeights.append(np.copy(x))
        self.weightsForUse = newWeights
        self.beta = beta
        self.lock.release()

    #This method performs a forward pass for the training method
    def getOutputs(self, inputs, beta):
        #add a column of ones to the inputs to account for the bias neurons
        inputsWithBias = np.concatenate((inputs, np.ones((len(inputs), 1))), axis=1)

        #calculate outputs for all inputs for the first layer
        outputs = np.array(np.dot(inputsWithBias, self.weights[0]))
        outputs = 1.0/(1.0 + np.exp(-outputs*beta))

        #calculate outputs for all other layers and append to the list
        outputs = [outputs]
        
        for x in range(len(self.structure)-2):
            outputs[x] = np.concatenate((outputs[x], np.ones((len(outputs[x]), 1))), axis=1)
            outputs.append(np.dot(outputs[x], self.weights[x+1]))
            outputs[x+1] = 1.0/(1.0 + np.exp(-outputs[x+1]*beta))

        return outputs

    #This method performs a forward pass for the robot to choose an action
    #It uses a separate weight array to prevent conflicts
    def getAction(self, state):
        
        #add a column of ones to the inputs to account for the bias neurons
        stateWithBias = np.concatenate(([state], np.ones((len([state]), 1))), axis=1)
        self.lock.acquire()
        #calculate outputs for all inputs for the first layer
        outputs = np.array(np.dot(stateWithBias, self.weightsForUse[0]))
        outputs = 1.0/(1.0 + np.exp(-outputs*self.beta))

        #calculate outputs for all other layers and append to the list
        outputs = [outputs]
        
        for x in range(len(self.structure)-2):
            outputs[x] = np.concatenate((outputs[x], np.ones((len(outputs[x]), 1))), axis=1)
            outputs.append(np.dot(outputs[x], self.weightsForUse[x+1]))
            outputs[x+1] = 1.0/(1.0 + np.exp(-outputs[x+1]*self.beta))
        self.lock.release()

        return outputs[-1][0]

