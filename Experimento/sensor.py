import os, time,FaBo9Axis_MPU9250,csv
import RPi.GPIO as GPIO

##buzzer inicialização

buzzer = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer, GPIO.OUT, initial=GPIO.LOW) 
GPIO.setwarnings(False)

mpu9250 = FaBo9Axis_MPU9250.MPU9250()

##buzzer sensor

for x in range (2):
    GPIO.output(buzzer,1)
    time.sleep(1)
    GPIO.output(buzzer,0)
    time.sleep(1)

## Determina um diretório inicial

directory = '/home/jupiterpi/Desktop/JunoIII/Dados/Sensor_' #Pasta em que são colocadas as fotos
n = 0
target_cam = directory + str(n)
flag = False

## Cria uma pasta nomeada no diretório determinado anteriormente
## Caso a pasta já exista no diretório, uma pasta com outro nome é criada

while flag != True:
    if os.path.exists(target_cam):
        n += 1
        target_cam = directory + str(n)
    else:
        os.mkdir(target_cam)
        with open(target_cam + '/DadosSensor_' + str(n) + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Indice','Ax','Ay','Az','Gx','Gy','Gz','Mx','My','Mz',])
        flag = True

#Função que lê os dados do sensor e salva eles em um csv
        
def sensor(instante):
    accel = mpu9250.readAccel()
    gyro = mpu9250.readGyro()
    mag = mpu9250.readMagnet()
    ax,ay,az = str(accel['x']),str(accel['y']),str(accel['z'])
    gx,gy, gz = str(gyro['x']),str(gyro['y']),str(gyro['z'])
    mx, my, mz = str(mag['x']), str(mag['y']), str(mag['z'])
    indice =str(i) + '_' + str(instante) 
    with open(target_cam + '/DadosSensor_' + str(n) + '.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([indice,ax,ay,az,gx,gy,gz,mx,my,mz,'\n'])

for i in range(100): 
    print(i)
    sensor(0)
    tstp = time.time()
    sensor(1)

def sensor2():
    accel = mpu9250.readAccel()
    gyro = mpu9250.readGyro()
    mag = mpu9250.readMagnet()
    ax,ay,az = str(accel['x']),str(accel['y']),str(accel['z'])
    gx,gy, gz = str(gyro['x']),str(gyro['y']),str(gyro['z'])
    mx, my, mz = str(mag['x']), str(mag['y']), str(mag['z'])
    indice =str(i) 
    with open(target_cam + '/DadosSensor_' + str(n) + '.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([indice,ax,ay,az,gx,gy,gz,mx,my,mz,'\n'])

for i in range(7200):
    sensor2()