"""
    This file serves as the main thread of control for the robot.It initializes, and acts as a buffer between the various other components.
"""

from queue import Queue
import threading
from datastore import DataStore
import random
import time
from Brain import Brain
from pi.motor2 import step
from pi.ultratest2 import read
from math import cos
from math import sin
from math import sqrt
from math import radians
from math import degrees
from math import pi
import sys
import pickle



chunkX = 0
chunkY = 0

jobBuffer = Queue()
readingBuffer = Queue()

#This method is initialized as a thread, and acts as the low level controller. It is responsible for reading from sensors, and carrying out the desired operations.
def sensorThread(jobs, readings):

    count = 0
    distSum = 0
    xCoord = 0
    yCoord = 0
    orientation = 0
    stepSize = 8
    
    direction = 120

    reading = read(direction)

    readings.put((reading[1], xCoord, yCoord, orientation))
    loopCounter = 0
    gotReading = False


    while True:
        #if the sensor has had time to take another reading get it
        if loopCounter == 15:
            reading = read(direction)
            loopCounter = 0
            gotReading = True
        else:
            gotReading = False
        loopCounter+=1 

        #if reading from the sensor succeeded 
        if gotReading:
            #If the reading is invalid disregard it
            if len(reading) < 2:
                print(":(")
                continue
            #otherwise record the observed distance
            dist = reading[1]
            distSum += dist
            count += 1

        gotReading = False
        time.sleep(0.001)

        #check for an action to perform
        try:
            job = jobs.get(True, 0.001)
        except:
            continue
        
        #Before moving place the currently read distance in the buffer
        #if multiple readings were performed they are averaged        
        while count > 0:
            try:
                readings.put((distSum/count, xCoord, yCoord, orientation), True, 0.001)
                distSum = 0
                count = 0
            except:
                print(sys.exc_info()[0])
                print("put reading failed")

        #perform the action
        if job[0] == 1 and job[1] == 1 and job[2] == 1 and job[3] == 1 and job[4] == 1:
            #This job indicates shutdown
            break
        elif job[0] == 1 and job[1] == 0 and job[2] == 0 and job[3] == 0:
            newPos = step('f', stepSize)
        elif job[0] == 0 and job[1] == 1 and job[2] == 0 and job[3] == 0:
            newPos = step('l', 8)
        elif job[0] == 0 and job[1] == 0 and job[2] == 1 and job[3] == 0:
            newPos = step('r', 8)
        elif job[0] == 0 and job[1] == 0 and job[2] == 0 and job[3] == 1:
            newPos = step('b', stepSize)
        else:
            newPos = (orientation, xCoord, yCoord)
               
        xCoord = newPos[1]
        yCoord = newPos[2]
           
        orientation = newPos[0]


sensorReader = threading.Thread(target=sensorThread, args=(jobBuffer, readingBuffer))

#channels for communicating with the dataStore
outChannel = Queue()
inChannel = Queue()

dataBank = DataStore(outChannel, inChannel, 100)

dataBank.start()
sensorReader.start()
print("started")

xCoord = 0
yCoord = 0

time.sleep(2)

#load the initial blank grid
grid = []
for x in range(-1, 2):
    grid.append([])
    for y in range(-1, 2):
        grid[x+1].append(dataBank.get(x, y))

xVect = 1
yVect = 0

oldxCoord = 0
oldyCoord = 0

chunkX = 0
chunkY = 0
oldChunkX = 0
oldChunkY = 0

#Although the datastore is capbale of generating a path from the robot to a node in need of exploration this feature has not been fully integrated. Difficulty training the neural network interfered with the integration of this feature.
path = []

NNInputSize = 10

score = 0
oldScore = 0
scores = []

action = [0,0,0,0,0]

dist = 0
#Initialize the Q-Learning algorithm with a learning rate of 0.2, discount = 0.99, and a seed of 44
NeuNet = Brain(10, 0.2, 0.99, 44)

NeuNet.start()

theWalk = []

#these are helper methods used to interact with the map
def getFromGrid(x, y):
    return grid[int(x/100)][int(y/100)][x%100][y%100]

def putToGrid(x,y,val):
    grid[int(x/100)][int(y/100)][x%100][y%100] = val

#The main control loop
someCtr = 0
while True:
    
    time.sleep(0.001)
    someCtr+= 1

    #exit after 1000 iterations
    print(someCtr)
    if someCtr > 1000:
        break
    change = 0
    try:
        #check for readings from the low level controller
        reading = readingBuffer.get(True, 0.001)
        
        dist = reading[0]

        #If the observation is reasonably close
        #this helps reduce noisy sensor readings
        #The minimum distance ensures that the system does not recieve a bonus to utility for driving into objects
        if dist < 50 and dist > 5:
            
            #compute a unit vector in the direction of the reading
            angle = radians(reading[3])
            print("angle: " + str(degrees(angle)))
            
            if (angle >= 0 and angle < pi/2) or angle == 360:
                x = cos(angle)*dist
                y = -sin(angle)*dist
            elif angle >= pi/2 and angle < pi:
                y = -cos(angle-(pi/2))*dist
                x = -sin(angle - (pi/2))*dist
            elif angle >= pi and angle < 3*pi/2:
                x = -cos(angle - pi)*dist
                y =  sin(angle - pi)*dist
            else:
                y = cos(angle-(3*pi/2))*dist
                x =  sin(angle-(3*pi/2))*dist
      
            if dist != 0:
                norm = sqrt((x*x) + (y*y))
            else:
                norm = 1
            x = x/norm
            y = y/norm
            xVect = x 
            yVect = y 
            a = 0
            b = 0
            origX = ((xCoord+50)%100) + 100
            origY = ((yCoord+50)%100) + 100
            
            oldScore = score
            
            #follow the unit vector through the grid, updating each point as appropriate
            while sqrt((a*a) + (b*b)) < dist:
                #This allows the width of the line of sight of the robot to be varied
                for i in range(0, 1):
                    #if we have exceeded the boundries of the grid break
                    if origX + a + i < 0 or origX + a + i >= 300 or origY + b < 0 or origY + b >= 300:
                        break
                    
                    else:
                        #otherwise updated the value at the current point
                        oldVal = getFromGrid(int(origX + a+i),int(origY + b))
                        newScore = (oldVal + (oldVal/1.5))/2
                        if oldVal > 1:
                            print("empty loop")
                            print("big oldVal")
                            break
                        if newScore > 1:
                            print("empty loop")
                            print("big new score")
                            break
                        if newScore < 0:
                            putToGrid(int(origX + a+i), int(origY + b), 0)
                        else:
                            putToGrid(int(origX + a+i), int(origY + b), newScore)
                #move one unit further along the vector
                a += x
                b += y
            
            change = 0
            #If we have not exceeded the bounds of the grid, mark the final position as an obstruction
            if a + origX >= 0 and a + origX < 300 and b + origY >=0 and b + origY < 300:
                
                oldVal = getFromGrid(int(a+origX), int(b+origY))
                newScore = (oldVal + ((oldVal+1)/2))/2

                if oldVal > 1:
                    print("big oldVal")
                    break
                if newScore > 1:
                    print("big new score")
                    break

                if newScore > 1:
                    putToGrid(int(a+origX),int(origY+b), 1)
                else:
                    putToGrid(int(a+origX),int(origY+b), newScore)
                
                
                change = getFromGrid(int(a+origX), int(b+origY)) - oldVal
             
                print(str(oldVal) + " " + str(getFromGrid(int(a+origX), int(b+origY))))

                #if the system was not confident in the contents of a point and the system has not become too close to an obstructed point update utility
                if change > 0.1:
                    obstructed = False
                    for p in range(-15, 15):
                        for q in range(-15, 15):
                            if getFromGrid(int(xCoord+p), int(yCoord+q)) > 0.5:
                                obstructed = True
                    if not obstructed:
                        score += change
                
        print("score: " + str(score))
            
        #Record the robots current position so that its path through space can be observed
        theWalk.append([xCoord, yCoord])
        
        oldxCoord = xCoord
        oldyCoord = yCoord

       
        xCoord = reading[1]
        yCoord = reading[2]

        oldChunkX = chunkX
        oldChunkY = chunkY

        #determine the coordinates of the chunk currently occupied by the robot
        #This is needed to interact with the data store
        if xCoord < 0:
            chunkX = int((xCoord-50)/100)
        else:
            chunkX = int((xCoord+50)/100)

        if yCoord < 0:
            chunky = int((yCoord-50)/100)
        else:
            chunkY = int((yCoord+50)/100)
        
    except:
        print("no readings")
        print(sys.exc_info())
        
    scores.append(score)
    
    print("X-Y: " + str(xCoord) + " " + str(yCoord))
    
    #If the robot has moved to a new chunk
    if chunkX != oldChunkX or chunkY != oldChunkY:
        
        #updated the grid, and add a connection to the data store
        if chunkY == oldChunkY:
            #if the robot has moved up 1 chunk
            if chunkX < oldChunkX:
                del grid[2]
                newRow = []
                newRow.append(dataBank.get(chunkX-1, chunkY-1))
                newRow.append(dataBank.get(chunkX-1, chunkY))
                newRow.append(dataBank.get(chunkX-1, chunkY+1))
                grid.insert(0, newRow)
                dataBank.connect((oldChunkX, oldChunkY), (chunkX,chunkY), [-1, 0])

            #else if the robot has moved down 1 chunk
            elif chunkX > oldChunkX:
                del grid[0]
                newRow = []
                newRow.append(dataBank.get(chunkX+1, chunkY-1))
                newRow.append(dataBank.get(chunkX+1, chunkY))
                newRow.append(dataBank.get(chunkX+1, chunkY+1))
                grid.append(newRow)
                dataBank.connect((oldChunkX, oldChunkY), (chunkX,chunkY), [1, 0])
        elif chunkY > oldChunkY:
            #if the robot has moved right and up 1 chunk
            if chunkX < oldChunkX:
                del grid[2]
                del grid[1][0]
                del grid[0][0]
                newRow = []
                newRow.append(dataBank.get(chunkX-1, chunkY-1))
                newRow.append(dataBank.get(chunkX-1, chunkY))
                newRow.append(dataBank.get(chunkX-1, chunkY+1))
                grid.insert(0, newRow)
                grid[1].append(dataBank.get(chunkX , chunkY+1))
                grid[2].append(dataBank.get(chunkX +1, chunkY+1))
                dataBank.connect((oldChunkX, oldChunkY), (chunkX,chunkY), [-1, 1])
            #else if the robot has moved right and down 1 chunk
            elif chunkX > oldChunkX:
                del grid[0]
                del grid[1][0]
                del grid[2][0]
                newRow = []
                newRow.append(dataBank.get(chunkX+1, chunkY-1))
                newRow.append(dataBank.get(chunkX+1, chunkY))
                newRow.append(dataBank.get(chunkX+1, chunkY+1))
                grid.append(newRow)

                grid[1].append(0,dataBank.get(chunkX, chunkY+1))
                grid[0].append(0,dataBank.get(chunkX-1, chunkY+1))
                dataBank.connect((oldChunkX, oldChunkY), (chunkX,chunkY), [1, 1])
            #else the robot has moved right 1 chunk
            else:
                del grid[0][0]
                del grid[1][0]
                del grid[2][0]
                grid[0].append(dataBank.get(chunkX-1, chunkY+1))
                grid[1].append(dataBank.get(chunkX, chunkY+1))
                grid[2].append(dataBank.get(chunkX+1, chunkY+1))
                dataBank.connect((oldChunkX, oldChunkY), (chunkX,chunkY), [0, 1])
        elif chunkY < oldChunkY:
            #if the robot has moved left and up 1 chunk
            if chunkX < oldChunkX:
                del grid[2]
                newRow = []
                newRow.append(dataBank.get(chunkX-1, chunkY-1))
                newRow.append(dataBank.get(chunkX-1, chunkY))
                newRow.append(dataBank.get(chunkX-1, chunkY+1))
                grid.insert(0, newRow)

                del grid[1][2]
                del grid[2][2]

                grid[1].insert(0, dataBank.get(chunkX, chunkY-1))
                grid[2].insert(0, dataBank.get(chunkX+1, chunkY-1))

                dataBank.connect((oldChunkX, oldChunkY), (chunkX,chunkY), [-1, -1])

            #else if the robot has moved left and down 1 chunk
            elif chunkX > oldChunkX:
                del grid[0]
                newRow = []
                newRow.append(dataBank.get(chunkX+1, chunkY-1))
                newRow.append(dataBank.get(chunkX+1, chunkY))
                newRow.append(dataBank.get(chunkX+1, chunkY+1))
                grid.append(newRow)

                del grid[0][2]
                del grid[1][2]
                grid[1].insert(0,dataBank.get(chunkX, chunkY-1))
                grid[0].insert(0,dataBank.get(chunkX-1, chunkY-1))
                dataBank.connect((oldChunkX, oldChunkY), (chunkX,chunkY), [1, -1])
            #else the robot has moved left 1 chunk
            else:
                del grid[0][2]
                del grid[1][2]
                del grid[2][2]
                grid[0].insert(0,dataBank.get(chunkX-1, chunkY-1))
                grid[1].insert(0,dataBank.get(chunkX, chunkY-1))
                grid[2].insert(0,dataBank.get(chunkX+1, chunkY-1))
                dataBank.connect((oldChunkX, oldChunkY), (chunkX,chunkY), [0, -1])
        
        oldChunkX = chunkX
        oldChunkY = chunkY

    #reduce the dimensionality of the grid so it will be suitable to use as input to the NN
    NNInput = [[0 for x in range(NNInputSize)] for y in range(NNInputSize)]
    chunkSize = int(300/NNInputSize)
    
    #reduce the grid to a 10x10 array where each cell is given the maximum value of all corresponding cells in the grid
    for a in range(len(grid)):
        for b in range(len(grid[a])):
            for x in range(NNInputSize):
                for y in range(10):
                    for row in range(NNInputSize):
                        val = max(grid[a][b][x+row][y:y+10])
                        cur = NNInput[int(((a*100)+(x*10))/chunkSize)][int(((b*100)+(y*10))/chunkSize)]
                        if val > cur:
                            NNInput[int(((a*100)+(x*10))/chunkSize)][int(((b*100)+(y*10))/chunkSize)] = val

    #Add markers to the neural network input indicating the position and orientation of the robot
    if xCoord + 150 < 0:
        xCoordInNNInput = 300 - ((xCoord+150)%300)
    else:
        xCoordInNNInput = (xCoord +150) % 300
    if yCoord +150 < 0:
        yCoordInNNInput = 300 - ((yCoord+150)%300)
    else:
        yCoordInNNInput = (yCoord +150) % 300
    xCoordInNNInput = int(xCoordInNNInput/30)
    yCoordInNNInput = int(yCoordInNNInput/30)
    NNInput[xCoordInNNInput][yCoordInNNInput] = 10

    if xVect > 0:
        xMod = 1
    elif xVect < 0:
        xMod = -1
    else:
        xMod = 0
    if yVect > 0:
        yMod = 1
    elif yVect < 0:
        yMod = -1
    else:
        yMod = 0
    if xCoordInNNInput + xMod < 10 and xCoordInNNInput + xMod >= 0:
        if yCoordInNNInput + yMod < 10 and yCoordInNNInput + yMod >= 0:
            NNInput[xCoordInNNInput+xMod][yCoordInNNInput+yMod] = 20
    
    
    #submit the input to the neural network to get an action
    action = NeuNet.getAction(score, NNInput) 

    #submit the action to the low level controller
    try:
        print(action)
        jobBuffer.put(action, True, 0.001)
    except:
        print("didnt put job")
        continue

#shut down the system
jobBuffer.put([1,1,1,1,1])
outChannel.put(["exit"])
NeuNet.shutdown()


print("ending")

#record information about this run
name = "run30"
aFile = open("paths/"+name+"walk", 'w')
s = ""

for x in theWalk:
    s += (str(x[0]) +","+str(x[1])+"\n")

print(s)
aFile.write(s)

aFile.close()

with open("paths/"+name+"map", 'wb') as outFile:
    cells = []
    for x in dataBank.chunks.keys():
        cells.append((x, dataBank.chunks[x].grid))
    pickle.dump(cells, outFile) 

with open("paths/"+name+"score", 'wb') as outFile:
    pickle.dump(scores, outFile)
    print(scores)

with open("paths/"+name+"error", 'wb') as outFile:
    pickle.dump(NeuNet.error, outFile)
    print(NeuNet.error)

