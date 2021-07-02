"""
    This class serves as the information store. It stores chunks that are currently not local to the robot, and is capable of pathfinding. It should be noted that path finding has not been fully integrated into the system. Difficulty training the neural network made pathfind capabilities unnecessary.
"""

import threading
from math import sqrt
import time
import pickle

class DataStore(threading.Thread):
    
    def __init__(self, inChannel, outChannel, size):
        threading.Thread.__init__(self)
        self.chunks = {}
        self.inChannel = inChannel
        self.outChannel = outChannel
        for x in range(-1, 2):
            for y in range(-1, 2):
                grid = [[0.5 for x in range(size)] for y in range(size)]
                self.chunks[(x,y)] = self.Node(grid, x,y)
        self.size = size

        self.xIndex = 0
        self.yIndex = 0

        self.worstScore = 1
        self.worstX = 0
        self.worstY = 0

        self.lock = threading.Lock()

    #This method is used to retrieve an chunk from the data store
    def get(self, x, y):
        self.lock.acquire()
        
        #If the chunk does not exist create it
        if (x, y) not in self.chunks:
            grid = [[0.5 for a in range(self.size)] for b in range(self.size)]
            self.chunks[(x, y)] = self.Node(grid, x, y)

        chunk = self.chunks[(x,y)].grid
        self.lock.release()
        return chunk

    #This method executes in a separate thread, and would be used for path finding if it had been integrated.
    def run(self):
        ctr = 0
        while True:
            #periodically write the current map to disk so it can be examined while the system continues to explore
            time.sleep(0.25)
            if ctr == 20:
                print("writing grid")
                with open("currentMap/map.dat", 'wb') as outFile:
                    self.lock.acquire()
                    cells = []
                    for x in self.chunks.keys():
                        cells.append((x,self.chunks[x].grid))
                    pickle.dump(cells, outFile) 
                    self.lock.release()
                ctr = 0
            
            ctr += 1
            
            #If a request for a path has been submitted
            try:
                item = self.inChannel.get(False, 0.01)
            except:
                continue

            #calculate the path and place it in the output buffer
            if item[0] == "path":
            
                thePath = self.findPath()
               
                try:
                    self.outChannel.put(thePath)
                except:
                    print("exception")
                
            elif item[0] == "exit":
                break

        print("datastore quit")    

    #This method is used to create a link between two adjacent nodes
    #Adjacent nodes are only linked when the robot travels between them, this ensures that a calculated path is viable 
    def connect(self, chunk1, chunk2, direction):
        self.lock.acquire()
        if direction[0] != 0 or direction[1] != 0:
            self.chunks[chunk1].links[(direction[0], direction[1])] = self.chunks[chunk2]
            self.chunks[chunk2].links[((-direction[0]), (-direction[1]))] = self.chunks[chunk1]
        self.lock.release()


    #This method runs an iterative deepending search to find a path
    def findPath(self):
        #continue until a path is found
        #At this time a path is guaranteed to exist since this method only finds a path from the robot to the node most in need of additional examination
        level = 1
        while True:
            #Unset all exploration flags
            for chunk in self.chunks.values():
                chunk.tested = False
          
            #perform a search to the current level
            path = self.testLevel(self.worstX, self.worstY, 0, level)

            #If no path is found try one level deeper, otherwise return the path
            if path is None or path:
                return path
            else:
                level += 1

    #This method performs a depth first from the lowest scored chunk to the robot, or the maximum level, whichever is reached first
    def testLevel(self, x, y, level, maxLevel):
        
        if level == maxLevel:
            return []

        #get the starting chunk
        chunk = self.chunks[(x,y)]
        
        #If the robot is in this chunk return 
        if x == self.xIndex and y == self.yIndex:
            return [(x,y)]

        #Set the tested flag and explore adjacent chunks
        chunk.tested = True
        
        for adjacent in chunk.links.items():
            
            if (adjacent[1] is not None) and (not adjacent[1].tested):
                
                step = self.testLevel(x + adjacent[0][0], y + adjacent[0][1], level+1, maxLevel)
            
                if step is not None:
                    if step:
                        step.append((x,y))
                        return step
                    else:
                        return []

        return None

    #This class is used to store chunks of the map
    class Node(object):
        def __init__(self, grid, x, y):
            self.update(grid)
            self.X = x 
            self.Y = y
            self.links = {(-1, 0):None, (1,0):None, (0,1):None, (0,-1):None, (-1,1):None, (-1,-1):None, (1,1):None, (1,-1):None}
            self.tested = False

            self.X = 0
            self.Y = 0
        
        #This method updates the current chunk
        def update(self, grid):
            self.grid = grid
            total = 0
            for x in grid:
                for y in x:
                    total += abs(0.5 - y)
            self.known = (total/(len(grid)*len(grid[0])))*2
