import random, csv

def sensor2():
    accel = random.random()
    gyro = random.random()
    mag = random.random()
    ax = str(accel)
    gx = str(gyro)
    mx = str(mag)
    indice = str(i) 
    with open('DadosSensor' + '.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([indice,ax,gx,mx])
        
for i in range(7200):
    sensor2()