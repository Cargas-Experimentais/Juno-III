import RPi.GPIO as GPIO
import time
buzzer = 4
GPIO.setmode(GPIO.BCM)

GPIO.setup(buzzer, GPIO.OUT, initial=GPIO.LOW) 
GPIO.setwarnings(False)
global buzz


for x in range (3):  
    GPIO.output(buzzer,1)
    print(1)
    time.sleep(1)
    GPIO.output(buzzer,0)
    print(0)
    time.sleep(1)

# buzz = GPIO.PWM(buzzer, 440)
# buzz.start(50)

# for x in range (3)
    




