import socket
import redis
import torch
import argparse

class DataMoverService():
    @staticmethod
    def connect_to_cache(cachelist: list) -> list:
        """implementation for redis cache"""
        cache_client_list = []
        for cacheip, port in cachelist:
            cache_client_list.append(
                redis.StrictRedis(host=cacheip, port=port, db=0))
            
    def start(self, batch_size: int, cachelist: list, ip: str,
              port: int, seqno:int, peerlist: list) -> None:
        self.batch_size = batch_size
        # first start your server at standard port, so trainer process can notify
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # bind it to port
        self.server_socket.bind((ip, port))
        # block until connection established
        self.server_socket.listen(1)
        # got connectino from trainer process
        self.trainer_socket, self.trainer_address = self.server_socket.accept()

        # start connection to cache (redis server)
        self.cache_client_list = DataMoverService.connect_to_cache(cachelist)
        # which cache to send
        self.sendcache_idx = (seqno + 1) % len(peerlist)
        # which cache to get
        self.getcache_idx = len(peerlist) - 1 if (seqno - 1) < 0  else seqno - 1
        self.seqno = seqno

    def updatecache(self, batch_no: int) -> None:
        for i in range(batch_no * self.batch_size, (batch_no + 1) * self.batch_size):
            # get
            databytes = self.cache_client_list[self.getcache_idx].get("data"+str(i))
            labelbytes = self.cache_client_list[self.getcache_idx].get("label"+str(i))
            # set
            # TODO:
            # check if we are getting byte sequence correctly
            self.cache_client_list[self.seqno].set("data"+str(i), databytes)
            self.cache_client_list[self.seqno].set("label"+str(i), labelbytes)

    def getcmd(self):
        while True:
            # recv from trainer
            # cmd size will be at most 16
            cmd = self.trainer_socket.recv(16).decode("utf-8")
            if cmd == "batch":
                batch_no = int.from_bytes(self.trainer_socket.recv(4), "little")
                self.updatecache(batch_no=batch_no)
                # update done send message
                self.trainer_socket.send(b"done")
            elif cmd == "exit":
                self.server_socket.close()
                break

class DataMoverServiceInterfaceClient():
    def __init__(self, ip: str, port: str):
        self.connection_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection_socket.connect((ip, port))
        self.first_time = True

    @staticmethod
    def convert_to_cmdbuffer(cmd_string: str) -> bytearray:
        cmdbytes = "batch".encode("utf-8")
        buffer = bytearray(16)
        buffer[:len(cmdbytes)] = cmdbytes
        return buffer

    def updatecache(self, batch_no: int) -> None:
        # recv from trainer
        # cmd size will be 16 byte
        # if not first time see if you have received any "done" message"
        if not self.first_time:
            cmd = ""
            while cmd != "done":
                cmd = self.connection_socket.recv(16).decode("utf-8")
        self.first_time = False

        # send the cmd
        self.connection_socket.send(
            DataMoverServiceInterfaceClient.convert_to_cmdbuffer("batch"))
        self.connection_socket.send(int.from_bytes(batch_no))
        
    def close(self):
        # send the cmd
        self.connection_socket.send(
            DataMoverServiceInterfaceClient.convert_to_cmdbuffer("exit"))
        self.connection_socket.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sno", "--seqno", type=int, help="which cache idx it will consider local", required=True)
    parser.add_argument("-bs", "--batch_size", type=int, help="batch size per iteration, default 16",
                        default=16)
    parser.add_argument("-cn", "--cache_nodes", nargs="+", type=list, help="how many redis cache node and their port", required=True)
    parser.add_argument("-pn", "--peer_nodes", nargs="+", type=list, help="how many peer trainer node and their port", required=True)
    parser.add_argument("-p", "--port", type=int, help="on which port service will listen", required=True)
    
    # get arguments
    args = parser.parse_args()

    cache_nodes = []
    for i in range(0, len(args.cache_nodes), 2):
        cache_nodes.append(("".join(args.cache_nodes[i]), "".join(args.cache_nodes[i+1])))
    print(cache_nodes)
    service = DataMoverService()
    # start
    service.start(
        batch_size=args.batch_size, cachelist=args.cache_nodes, ip="127.0.0.1", port=args.port,
        seqno=args.seqno, peerlist=args.peer_nodes
    )
    service.getcmd()




