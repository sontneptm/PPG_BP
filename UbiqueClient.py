from socket import *
import threading
import json
import time
import serial

#from django.conf import global_settings
global serverIp
global serverPort
global clientSock
global readySocket
global dict1
global dict2
global dict3
global message
global fd_port
global data_all_wave
global data_all_br
global wave
dict1 = {'jsm' : 178 , 'ccw' : 169, 'pms' : 168}
dict2 = {'성다연' : 168, '이예원' : 161, '백정은' :160}
testJson = {'msg_type' : 'child_register', 'msg_content' : {'id' : '2', 'name': 'conqueror', 'number': '01031064048'}}
readySocket = 0
serverPort = 5000

#serverIp = "220.69.203.14" #정은
#serverIp = "220.69.203.27" #다연
serverIp = "220.69.203.88" #철우
#서버에 접속하는 함수
def connectServer(ip, port) :
	global clientSock
	clientSock = socket(AF_INET, SOCK_STREAM)
	clientSock.connect((ip, port))
	print('연결 확인 됐습니다.')

	th = threading.Thread(target=receive_data, args=())
	th.start()
	
	

#데이터를 수신할 스레드
def receive_data() :
	global clientSock
	while True :
		data = clientSock.recv(1024).decode()
		print("데이타 : " + str(data))

#데이터 송신
def transmitData(data_all_wave, data_all_hr) :
	global clientSock
	global message
	#print('이것의 데이터 타입!!!' ,type(data_all_wave))
	dictionaries_wave={ 'msg_type' : 'waves', 'msg_content' : data_all_wave}
	dictionaries_hr={ 'msg_type' : 'hr', 'msg_content' : data_all_hr}

	try:
		message = json.dumps(dictionaries_wave)
		message2 = json.dumps(dictionaries_hr)
		
		print(message)
		print('send!!!!!')
		print(message2)
	except(TypeError, ValueError) :
		raise('You can only send JSON-serializable data')
		

	clientSock.send(message.encode('utf-8'))
	time.sleep(0.01)
	clientSock.send(message2.encode('utf-8'))  

def conv_value(n):
	val = (n & 0xf0)
	val = val>>4
	val = val*10 + (n & 0x0f)
	return val

def real_open_serial():
	
	
	BAUDRATE = 38400
	ser = serial.Serial(port = "/dev/ttyUSB0", baudrate=BAUDRATE, timeout=2)
	
	global data_all_wave
	data_all_wave = []
	global wave
	wave=0
	global data_all_hr
	data_all_hr = []
	global hr
	hr=0
	
	if (ser.isOpen() == False):
		ser.open()
	
	while True : 
		
		ser.flushInput()
		ser.flushOutput()
		data = ser.read(1)
		
		
		if data == bytearray([250]) :
			
			#print("hello")
			time.sleep(0.001)
			
			data = ser.read(10)
			#print("===================================================")
			if len(data) > 0 :
				
				#str = data.decode("utf-8")
				
				wave=conv_value(data[0])*100 + conv_value(data[1])
				hr = conv_value(data[2])*100 + conv_value(data[3])
				spo2 = conv_value(data[4])*100 + conv_value(data[5])
				bardata=conv_value(data[6])
				wave_gain=conv_value(data[7])
				status=conv_value(data[8])
				
				"""
				print('wave')
				print(wave)
				print('hr')
				print(hr)
				print('spo2')
				print(spo2)
				print('bardata')
				print(bardata)
				print('wave_gain')
				print(wave_gain)
				print('status')
				print(status)
				"""
				
				strs_wave=''		
				strs_hr=''	
				data_all_wave.append(wave)
				data_all_hr.append(hr)
		
		if len(data_all_wave)==50:
			
			for i in data_all_wave:
				strs_wave=strs_wave+','+str(i)
			for j in 	data_all_hr:
				strs_hr=strs_hr+','+str(j)
			
			print('thread start~~~~~~~~~~~~~~~')
			
			sendth = threading.Thread(target=transmitData(strs_wave,hr), args=())
			sendth.start()
			data_all_wave.clear()
			data_all_hr.clear()
	
			
		
def main():
	
	connectServer(serverIp, serverPort)
	real_open_serial()

main()
