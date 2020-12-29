import ppg.communication_module as cm
#import database_module as dm
import threading
import time
import json

class module_communicator:

    def __init__(self, server):
        self.num_list = []
        self.split_list_all= []
        self.count = 0
        self.server = server
        self.handler = threading.Thread(target=self.run, args=())
        self.handler.start()
        self.is_file_make = True #파일을 만들것이라면 True 아니면 False
        self.maked_file =  self.maked_file = open('read_ppg.csv', 'a') # a = 새로운 내용 추가, w = 새로운 파일 쓰기, r = 파일 내용 읽기 (끝나고 close 해주어야 함)

    def run(self):
        # 무한 루프로 돌려버립니다.
        while True:
            # 클라이언트 목록을 향상된 for문으로 참조합니다.
            for c in self.server.client_handler_list:
                # 만일 해당 클라이언트의 리퀘스트가 NO REQUEST가 아니라면
                if c.request != cm.order_enum.NO_REQUEST:
                    try:
                        if c.request == cm.order_enum.JUST_MESSAGE:
                            content = c.recv_msg.get('msg_content')
                            self.just_message(content)
                        elif c.request == cm.order_enum.EXCUTE_QUERY:
                            content = c.recv_msg.get('msg_content')
                            self.db.excute_query(content)
                        elif c.request == cm.order_enum.LIST_REQUEST:
                            self.list_request(c)
                        elif c.request == cm.order_enum.WAVES:
                            content = c.recv_msg.get('msg_content')
                            self.go_android(c.recv_msg, content)
                            #self.just_message(content) #######처리하는 코드로바꾸기
                            #self.recv_msg = self.sock.recv(1024).decode("utf-8")
                        elif c.request == cm.order_enum.BP:
                            content = c.recv_msg.get('msg_content')
                            self.go_android2(c.recv_msg, content)
                        else:
                            print("unknown request :" + c.request)
                    except Exception as e:
                        c.error_msg = str(e)
                        print(c.error_msg)
                    finally:
                        c.request = cm.order_enum.NO_REQUEST
            time.sleep(0.01)
        # end of while loop
        #clear_all(self.db, self.server)

    #ppg에서 받은 bp를 여기를 거쳐서 인공지능 모듈로
    def go_android2(self, content, content_get):
        for client in self.server.client_handler_list:
            dictionaries2={'msg_type':'bp','msg_content':'120,80'}

            try:
                message2=json.dumps(dictionaries2)
            except:
                raise ("dddddddd")

            client.sock.sendall(message2.encode("utf-8"))

    #waves를 안드로이드로 보냄
    def go_android(self, content, content_get):
        for client in self.server.client_handler_list:
            dictionaries={'msg_type':'waves','msg_content':content_get}

            try:
                message=json.dumps(dictionaries)
            except:
                raise ("dddddddd")

            client.sock.sendall(message.encode("utf-8"))

            '''
            client.sock.sendall(json.dumps({'msg_type':'waves','msg_content':content_get})).encode("utf-8")
            content = json.dumps(content)
            client.sock.sendall(content.encode("utf-8"))
            '''

        #200개 넘으면 인공지능 모듈에 보낼거기때문에 함수 실행

        #여기서 split해서 10개씩 끊어진 str를 하나의 리스트로 만들었음
        #,로 시작하기 때문에 [0]는 무조건 빈 값이 들어가므로 삭제해줬음!
        #200개를 세기 위해 count 추가해서 확인하도록 했음
        content_get_split = content_get.split(',')
        del content_get_split[0]
        self.count = self.count + 1

        #self.count가 1개당 list 안에 데이터는 10개! 그러므로 200의 데이터 리스트를 모으려면 count가 20가 될때까지 계속 더해줘서
        #200개의 데이터가 모여있는 1개의 리스트: self.split_list_all
        #count가 20개일때 인공지능 모듈로 보내주고, 함수 실행하라고 했지만.. 다연이는 말 안드러ㅓㅓㅓ
        if self.count<=4:
            self.split_list_all = self.split_list_all + content_get_split

        if self.count == 4:
            #print(self.split_list_all)
            bp_thread = threading.Thread(target=self.send_blood_pressure, args=(self.split_list_all, ))
            bp_thread.start()
            #
            if self.is_file_make :
                data_save_thread = threading.Thread(target=self.save_ppg_data, args=(self.split_list_all, ))
                data_save_thread.start()
            self.count = 0
            self.split_list_all = []

    def save_ppg_data(self, ppg_wave_list):
        ppg_data_str = str(ppg_wave_list[0])
        for i in range(1, len(ppg_wave_list)) :
            ppg_data_str = ppg_data_str + "," + ppg_wave_list[i]
        self.maked_file.write(str.format(ppg_data_str) + "\n")

        print("saving...")


    def send_blood_pressure(self, list):
        #인공지능 모듈로 예측
        #a = result(list)
        blood_pressure = "120,80"
        bp_dict = {'msg_type' : 'bp', 'msg_content' : blood_pressure}
        print(list)
        try :
            for c in self.server.client_handler_list:
                print(bp_dict)
                c.sock.sendall(json.dumps(bp_dict).encode("utf-8"))
        except Exception as e :
            print('혈압 송신에서 예외 발생', e)



    def just_message(self, content):
        print(content)

    def list_request(self, c):
        print("리스트 리퀘스트")
        content = c.recv_msg.get('msg_content')
        list_type = c.recv_msg.get('list_type')
        datas = self.db.select_query(content)
        print(datas)

        json_array = {'msg_type': 'data_set'}
        json_array_list = []
        json_list = []

        j = 0;
        for i in datas:
            if list_type == "car" :
                dict = {'car_id' : i[0], 'car_number':i[1], 'car_type':i[2], 'car_description':i[3]}
            elif list_type == "passer" :
                dict = {'passer_id': i[0], 'passer_name': i[1], 'passer_duty': i[2], 'rank_name': i[3]}
            elif list_type == "record" :
                dict = {'enter_record_id': i[0], 'enter_type': i[1], 'car_number': i[2], 'passer_name': i[3], 'rank_name':i[4], 'enter_purpose':i[5], 'record_time': str(i[6]) }

            json_list.append(dict)
            j += 1

            if j % 5 == 0 :
                j = 0
                json_array['msg_content'] = json_list
                json_array_list.append(json_array)
                json_array = {'msg_type': 'data_set'}
                json_list = []

        if j != 0 :
            json_array['msg_content'] = json_list
        json_array_list.append(json_array)
        print(json_array_list)


        clear_message = json.dumps({'msg_type':'db_clear'})
        c.sock.sendall(clear_message.encode("utf-8"))
        time.sleep(0.05)

        for i in json_array_list:
            i['list_type'] = list_type
            lists = json.dumps(i)
            c.sock.sendall(lists.encode("utf-8"))
            time.sleep(0.05)

        finish_message = json.dumps({'msg_type':'finish_data'})
        c.sock.sendall(finish_message.encode("utf-8"))

# end of funtion
# end of class
def clear_all(server, db):  # video stream will be terminated automatically
    server.clear()
    db.clear()

def main():

    server = cm.server_class()
    module_communicator( server = server)

if __name__ == "__main__":
    main()
