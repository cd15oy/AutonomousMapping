"""
This file provides a method to controller the stepper motors on the robot.
Note that the library gpiozero is used to interact with the Raspberry Pi's gpio pins. More information about this library and its maintainers can be found at https://gpiozero.readthedocs.io/en/stable/
"""

from gpiozero import OutputDevice

import time

from math import cos
from math import sin
from math import pi
from math import radians
from math import degrees

#Define the the pins to control the motors
RMotor = [OutputDevice(6), OutputDevice(13), OutputDevice(19), OutputDevice(26)]
LMotor = [OutputDevice(17), OutputDevice(4), OutputDevice(22), OutputDevice(27)]

#These are two helper methods used to set pairs of pins which follow a reliable pattern
def high(line):
	line[0].on()
	line[1].on()

def low(line):
	line[0].off()
	line[1].off()

#variables to track the robots position relative to its origin
orientation = 0
xCoord = 0
yCoord = 0

#The distance traveled with each step
forwardStep = (pi*5.04)*(1.0/400)
#The change in orientation with each turn in place step
turnStep = (forwardStep/(pi*11.5))*360
"""
Unfortunately the constants above must vary considerably depending on the surface, battery power, and other real world factors
"""

lastDir = 'q'

#This method provides the required pattern of input to the motors in order for the motors to turn
#direction is a character either 'f', 'b', 'r', or 'l'
#count is the number of steps to perform
def step(direction, count) :
	global lastDir
	#pause slightly between changing directions to prevent the robot from slipping
	if lastDir != direction:
		lastDir = direction
		time.sleep(0.08)
	
	#travel slightly slower when turning
	if direction == 'r' or direction == 'l':
		sleepTime = 0.008
	else:
		sleepTime = 0.005

	#The following statements modify the order in which motor inputs will be set and released to produce the desired movement
	if direction == "b":
		one = [RMotor[0], LMotor[0]]
		two = [RMotor[1], LMotor[1]]
		three = [RMotor[2], LMotor[2]]
		four = [RMotor[3], LMotor[3]]
	
	elif direction == "f" :
		one = [RMotor[1], LMotor[1]]
		two = [RMotor[0], LMotor[0]]
		three = [RMotor[3], LMotor[3]]
		four = [RMotor[2], LMotor[2]]

	elif direction == "l":
		one = [RMotor[0], LMotor[1]]
		two = [RMotor[1], LMotor[0]]
		three = [RMotor[2], LMotor[3]]
		four = [RMotor[3], LMotor[2]]

	elif direction == "r":
		one = [RMotor[1], LMotor[0]]
		two = [RMotor[0], LMotor[1]]
		three = [RMotor[3], LMotor[2]]
		four = [RMotor[2], LMotor[3]]
	
	else:
		return None

	#Step until the step count has been reached
	stepCount = 0
	while True:
		#ensure the correct initial state
		low(one)
		low(two)
		high(three)
		high(four)

		#A number of sleep statements give the coils in the motors time to charge
		time.sleep(sleepTime)
		
		high(two)
		
		time.sleep(sleepTime)
		
		stepCount+=1
		if stepCount == count:
			break
		
		low(three)

		time.sleep(sleepTime)

		stepCount+=1
		if stepCount == count:
			break

		high(one)
		
		time.sleep(sleepTime)

		stepCount+=1
		if stepCount == count:
			break
		
		low(four)

		time.sleep(sleepTime)

		stepCount+=1
		if stepCount == count:
			break

		high(three)
		
		time.sleep(sleepTime)

		stepCount+=1
		if stepCount == count:
			break
		
		low(two)

		time.sleep(sleepTime)

		stepCount+=1
		if stepCount == count:
			break

		high(four)
		
		time.sleep(sleepTime)

		stepCount+=1
		if stepCount == count:
			break;		
		
		low(one)

		time.sleep(sleepTime)
		
		stepCount+=1
		if stepCount == count:
			break


	#calculate the change in orientation and postion
	global orientation
	global xCoord
	global yCoord		
	if direction == 'l':
		orientation -= (turnStep*count)
		if orientation < 0:
			orientation += 360
	elif direction == 'r':
		orientation += (turnStep*count)
		if orientation > 360:
			orientation -= 360
	elif direction == 'f':
		xCoord += cos(radians(orientation))*forwardStep*count
		yCoord += sin(radians(orientation))*forwardStep*count
	elif direction == 'b':
		xCoord += cos(radians(orientation + 180))*forwardStep*count
		yCoord += sin(radians(orientation+180))*forwardStep*count
				
	return (orientation, xCoord, yCoord)
