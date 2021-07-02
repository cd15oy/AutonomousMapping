"""
	This script provides a method for reading from the ultrasonic sensor. 
"""
import serial
import time 
from math import cos
from math import sin
from math import sqrt

#Initialize the serial port, the UART lines
port = serial.Serial("/dev/serial0", baudrate=9600, timeout=1.0, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS, xonxoff=True)


command = [0x22,0x0a,0x00,0x2c]

#This method will read the distance
#Since the ultrasonic provides a servo controller the direction parameter can be used to issue commands to the servo
#direction should be a value in [0,180]
def read(direction):

	global port
	
	if command[1] != int(direction/6):
		command[1] = int(direction/6)
		command[3] = command[0] + command[1]
		port.write(bytes(command))
		#if the servo us not currently at the desired direction move it first, and then read the distance
		#otherwise just take a distance reading
	
	port.write(bytes(command))
	tmp = port.read(4)

	#If the sensors stops responding reinitialize the connection
	if len(tmp) < 4:
		port.close()
		time.sleep(2)
		port = serial.Serial("/dev/serial0", baudrate=9600, timeout=1.0, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS, xonxoff=True)
		return []
	else:
		return (command[1], (tmp[1]<<8)+tmp[2])
