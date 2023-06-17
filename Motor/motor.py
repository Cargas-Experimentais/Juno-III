from machine import Pin
import time

step = 2
dir = 3

file = open("direction.csv", "r")

direction = int(file.read())
print(direction)

file.close()
file = open("direction.csv", "w")

if direction == 1:
    file.write(str(0))
else:
    file.write(str(1))

pinstep = Pin(step, Pin.OUT)
pindir = Pin(dir, Pin.OUT)

duration = 150

pindir.value(direction)

for i in range(duration):
    pinstep.value(1)
    time.sleep(0.01)
    pinstep.value(0)
    time.sleep(0.01)

file.close()