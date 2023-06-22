from picamera2 import Picamera2, Preview
import os,time, datetime
import RPi.GPIO as GPIO

##buzzer inicialização
time.sleep(5)
buzzer = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer, GPIO.OUT, initial=GPIO.LOW) 
GPIO.setwarnings(False)

## Determina um diretório inicial

directory = '/home/jupiterpi/Desktop/JunoIII/Dados/Imagens_' #Pasta em que são colocadas as fotos
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
        flag = True

## Configurações da camera
        
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1000, 750)}, lores={"size": (640, 480)}, display="lores")
picam2.configure(camera_config)
picam2.start_preview(Preview.QTGL)
picam2.start()
time.sleep(2)

## Buzzer de camera

for x in range (3):
    GPIO.output(buzzer,1)
    time.sleep(0.5)
    GPIO.output(buzzer,0)
    time.sleep(0.5)
    

## Loop que tira todas as fotos

for i in range(7200): 
    print(i)
    time.sleep(1)
    ct = datetime.datetime.now()
    ts = ct.timestamp()
#     tstp = time.time()
    picam2.capture_file(target_cam + '/' + str(int(ts)) + '.jpg') # Nome da foto = Imagem_ + i
    
picam2.stop_preview()

