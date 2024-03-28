import socket
import struct
import yaml
import redis
import time
import argparse

import mindist_chain


class DataMoverService():
    MAXCMD_SIZE = 16
    INTSIZE = 4
    FLOATSIZE = 8

    def connect_to_cache(self):
        """implementation for redis cache"""
        for cacheid in self.cache_node_dict:
            # the cache may come back online at delay
            # keep trying for 100s
            connection_refused_count = 0
            while connection_refused_count < 100:
                try:
                    cacheip = self.cache_node_dict[cacheid][0]
                    cacheport = self.cache_node_dict[cacheid][1]
                    self.cache_node_dict[cacheid].append(
                        redis.StrictRedis(host=cacheip, port=cacheport, db=0))
                    break
                except ConnectionError as e:
                    connection_refused_count += 1
                    print("connection establish attempt {0} to {1}:{2} failed at cahche no {3}".format(
                        connection_refused_count, cacheip, cacheport, self.seqno))
                    # sleep for a second
                    time.sleep(1)
            
    def collect_latency_data(self):
        # how to collect?
        # we will query for min(64, cache length) data unit 10 times
        # above choices are arbitrary
        # we are not selecting same data unit multiple times to emulate the 
        # random query incoming to cache this behavior
        latency_data = [0] * len(self.cache_node_dict)
        for cacheid in self.cache_node_dict:
            redis_client = self.cache_node_dict[cacheid][5]
            redis_client_size = self.cache_node_dict[cacheid][3]

            # query loop
            query_time_start = time.time()
            total_byte_read = 0
            for i in range(64):
                databytes = redis_client.get("data"+str(i%redis_client_size))
                labelbytes = redis_client.get("label"+str(i%redis_client_size))
                total_byte_read += len(databytes) + len(labelbytes)
            # latency per byte
            latency_data[cacheid] = (time.time() - query_time_start) / total_byte_read

        return latency_data
    
    def send_latency_data(self, connection_socket, latency_data_list):
        # first send self id/seqno
        connection_socket.send(int.to_bytes(self.seqno, length=4, byteorder="little"))
        # then send  number of peer of whom sending the latency data
        connection_socket.send(int.to_bytes(len(latency_data_list), length=4, byteorder="little"))
        # send all the latency data
        for latency in latency_data_list:
            connection_socket.send(struct.pack('d', latency))

    def create_latency_matrix(self, peer_connection_list, latency_data_list) -> list[list[float]]:
        latency_matrix = [latency_data_list]
        
        # now collect data from each peer socket and fill up latency matrix
        # 0 is the collector peer, so no need to read from itself
        print("receiving latency information")
        for i in range(1, len(peer_connection_list)):
            latency_matrix.append([])
            peer_socket = peer_connection_list[i]
            peercount = int.from_bytes(peer_socket.recv(4), "little")
            for j in range(peercount):
                latency = struct.unpack('d', peer_socket.recv(8))[0]
                latency_matrix[i].append(latency)

        return latency_matrix
    
    def send_chain_data(self, chain, peer_connection_list):
        # find the collectors ip and service port
        for i in range(0, len(peer_connection_list)):
            peer_socket = peer_connection_list[i]
            # find position in chain
            for pos, peerid in enumerate(chain):
                if peerid == i:
                    cacheupdate_source_nodeid = chain[(pos+1)%len(chain)]
                    if i != 0:
                        # close that redis connection
                        self.cache_node_dict[cacheupdate_source_nodeid][5].close()
                        # send the data
                        peer_socket = peer_connection_list[i]
                        peer_socket.send(int.to_bytes(cacheupdate_source_nodeid, length=4, byteorder="little"))
                    else:
                        self.getcache_idx = cacheupdate_source_nodeid

    def get_chain_data(self, connection_socket) -> int:
        return int.from_bytes(connection_socket.recv(4), "little")

    def global_sequence_sync(self, self_latency_data):
        # seqno 0 has the responsiblity to collect latency data and decide the chain
        # and then send each of them the data
        if self.seqno != 0:
            # 20s wait to give seqno 0 some chance to setup stuffs
            print("waiting before sending data")
            time.sleep(10)
            print("connecting with cache seq 0 service")
            # find the collectors ip and service port
            ip = self.cache_node_dict[0][0]
            serviceport = self.cache_node_dict[0][4]
            # create connection
            # it will try for 100s
            connection_refused_count = 0
            while connection_refused_count < 100: 
                try:
                    connection_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    connection_socket.connect((ip, int(serviceport)))
                    break
                except ConnectionError as e:
                    connection_refused_count += 1
                    print("connection establish attempt to {0} failed at cahche no {1}".format(connection_refused_count, ip, serviceport, self.seqno))
                    # sleep for a second
                    time.sleep(1)
            
            print("sending latency data")
            self.send_latency_data(connection_socket=connection_socket, latency_data_list=self_latency_data)
            print("getting chain data")
            self.getcache_idx = self.get_chain_data(connection_socket=connection_socket)
            print("sync with cache 0 is done, closing connection...")
            connection_socket.close()
            print("closing extra opened redis connections")
            # we need only the seq no and getcache_idx
            for i in self.cache_node_dict:
                if i == self.seqno or i == self.getcache_idx:
                    continue
                self.cache_node_dict[i][5].close()
        else:
            peer_connection_list = [None] * len(self.cache_node_dict)
            # there will be n-1 connection if there are total n nodes
            print("accepting peer connection to get latency information")
            for i in range(len(self.cache_node_dict) - 1):
                self.server_socket.listen(1)
                # got connectino from trainer process
                peer_socket, _ = self.server_socket.accept()
                # first recv the peer seqno
                peerseqno = int.from_bytes(peer_socket.recv(4), "little")
                print("peer {0} has connected to do sync".format(peerseqno))
                peer_connection_list[peerseqno] = peer_socket
            # collecting latency data
            print("collecting latency data")
            latency_matrix = self.create_latency_matrix(peer_connection_list, self_latency_data)
            # generate mindist chain
            print("generating sequence")
            chain = mindist_chain.generate_mindistchain(distmatrix=latency_matrix)

            # now send the chain information to all
            print("sending chain information to all")
            self.send_chain_data(chain=chain, peer_connection_list=peer_connection_list)

        print("global sync step is done")

    def parse_cachedesc(self, cachefilepath: str):
        with open(cachefilepath) as fin:
            datadict = yaml.safe_load(fin)

        datadict = datadict["cachedict"]
        cache_nodes_dict = {}
        rank_id_dict = {}
        for i, key in enumerate(datadict):
            rank = key
            rank_id_dict[i] = rank

            ip = datadict[key][0].split(":")[0]
            port = datadict[key][0].split(":")[1]
            offset = datadict[key][1]["offset"]
            length = datadict[key][1]["length"]
            serviceport = datadict[key][1]["serviceport"]
            cache_nodes_dict[i] = [ip, port, offset, length, serviceport]

        return cache_nodes_dict

    def start(self, cachedesc_filepath: str, seqno:int) -> None:
        self.batch_size = 1
        self.getcache_idx = 0
        # sequence no
        self.seqno = seqno

        # parse the file for global info on the caches
        self.cache_node_dict = self.parse_cachedesc(cachefilepath=cachedesc_filepath)
        # get self ip
        self.ip = self.cache_node_dict[seqno][0]
        # get the port where every one will connect or you will connect to everyone
        self.port = self.cache_node_dict[seqno][4]
        # first start your server at standard port, so trainer process can notify
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # bind it to port
        self.server_socket.bind((self.ip, self.port))


        # start connection to cache (redis server)
        print("creating client to each cache")
        self.connect_to_cache()
        print("collecting latency data to each cache")
        latency_data = self.collect_latency_data()

        print("proceeding to sync globally about the latency")
        self.global_sequence_sync(self_latency_data=latency_data)

        print("starting interface to client")
        # block until connection established
        self.server_socket.listen(1)
        # got connectino from trainer process
        self.trainer_socket, self.trainer_address = self.server_socket.accept()

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

    def cmdloop(self):
        while True:
            # recv from trainer
            # cmd size will be at most DataMoverService.MAXCMD_SIZE
            cmd = self.trainer_socket.recv(DataMoverService.MAXCMD_SIZE).decode("utf-8")
            if cmd[:2] == "bs":
                batch_size = int.from_bytes(self.trainer_socket.recv(4), "little")
                self.batch_size = batch_size
                # update done send message
                self.trainer_socket.send(b"done")
            elif cmd[:5] == "batch":
                batch_no = int.from_bytes(self.trainer_socket.recv(4), "little")
                self.updatecache(batch_no=batch_no)
                # update done send message
                self.trainer_socket.send(b"done")
            elif cmd[:4] == "exit":
                self.server_socket.close()
                break

class DataMoverServiceInterfaceClient():
    def __init__(self, ip: str, port: str):
        self.connection_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection_socket.connect((ip, int(port)))
        self.first_time = True

    @staticmethod
    def convert_to_cmdbuffer(cmd_string: str) -> bytearray:
        cmdbytes = cmd_string.encode("utf-8")
        buffer = bytearray(DataMoverService.MAXCMD_SIZE)
        buffer[:len(cmdbytes)] = cmdbytes
        return buffer

    def updatecache(self, batch_no: int) -> None:
        # recv from trainer
        # cmd size will be DataMoverService.MAXCMD_SIZE byte
        # if not first time see if you have received any "done" message"
        if not self.first_time:
            cmd = ""
            while cmd[0:4] != "done":
                cmd = self.connection_socket.recv(4).decode("utf-8")
        self.first_time = False

        # send the cmd
        self.connection_socket.send(
            DataMoverServiceInterfaceClient.convert_to_cmdbuffer("batch"))
        self.connection_socket.send(int.to_bytes(batch_no, length=4, byteorder="little"))

    def set_batchsize(self, batch_size:int) -> None:
        # recv from trainer
        # cmd size will be DataMoverService.MAXCMD_SIZE byte
        # if not first time see if you have received any "done" message"
        if not self.first_time:
            cmd = ""
            while cmd[0:4] != "done":
                cmd = self.connection_socket.recv(4).decode("utf-8")
        self.first_time = False

        # send the cmd
        self.connection_socket.send(
            DataMoverServiceInterfaceClient.convert_to_cmdbuffer("bs"))
        self.connection_socket.send(int.to_bytes(batch_size, length=4, byteorder="little"))
        
    def close(self):
        # this check is done to confirm final update is done
        # TODO
        # create a pending status and pending_expected message attribute
        # don't execute any next status until pending is cleared with expected message
        if not self.first_time:
            cmd = ""
            while cmd[0:4] != "done":
                cmd = self.connection_socket.recv(4).decode("utf-8")
        # send the cmd
        self.connection_socket.send(
            DataMoverServiceInterfaceClient.convert_to_cmdbuffer("exit"))
        self.connection_socket.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sno", "--seqno", type=int, help="which cache idx it will consider local", required=True)
    parser.add_argument("-cdesc", "--cache-descriptor", type=str, help="yaml file describing caches", default="cache.yaml", required=False)

    # get arguments
    args = parser.parse_args()

    service = DataMoverService()
    # start
    service.start(
        cachedesc_filepath=args.cache_descriptor, seqno=args.seqno
    )
    service.cmdloop()




