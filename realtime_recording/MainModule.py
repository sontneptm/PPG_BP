import PykinectRealTime.communication_module as cm
import threading
import time
import json

class module_communicator:
    def __init__(self, server):
        self.server = server
        self.handler = threading.Thread(target=self.run, args=())
        self.handler.start()
        self.send_sitting_threads = threading.Thread(target=self.send_sitting_thread, args=())
        self.send_sitting_threads.start()

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
                            print(content)
                            self.just_message(content)
                        elif c.request == cm.order_enum.EXCUTE_QUERY:
                            content = c.recv_msg.get('msg_content')
                            self.db.excute_query(content)
                        elif c.request == cm.order_enum.LIST_REQUEST:
                            self.list_request(c)
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

    def send_sitting_thread(self):
        while True :
            for i in range(0,5):
                self.send_sitting_posture(i)
                time.sleep(1)


    def send_sitting_posture(self, sit_num):
        sit_message = {'msg_type' : 'sit_posture', 'msg_content' : sit_num}
        sit_message = json.dumps(sit_message).encode('utf-8')

        for c in self.server.client_handler_list:
            c.sock.sendall(sit_message)


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
