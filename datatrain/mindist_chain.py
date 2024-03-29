import yaml

def generate_mindistchain(distmatrix):
    # Find the edge with minimum length and init the chain
    chain = []
    minedge_dist = max([max(row) for row in distmatrix])
    for i in range(len(distmatrix)):
        for j in range(len(distmatrix)):
            if i!=j and distmatrix[i][j] < minedge_dist:
                chain = [i,j]
                minedge_dist = distmatrix[i][j]

    # create remaining node list
    rem_node_list = []
    for i in range(len(distmatrix)):
        if i not in chain:
            rem_node_list.append(i)

    while len(rem_node_list) > 0:
        added_node_idx = 0
        # find the node which will increase current chain minimally
        # initialization for that
        min_inc = distmatrix[chain[0]][rem_node_list[0]] + distmatrix[chain[-1]][rem_node_list[0]]
        added_node_idx = 0
        final_inserted_pos = 0
        for node_idx in rem_node_list:
            # increment calc
            for pos in range(len(chain)):
                increment = distmatrix[chain[pos]][node_idx] + distmatrix[chain[pos-1]][node_idx]
                if min_inc > increment:
                    final_inserted_pos = pos
                    added_node_idx = node_idx
        
        chain.insert(final_inserted_pos, rem_node_list[added_node_idx])
        print(chain)
        rem_node_list.pop(added_node_idx)

    return chain

if __name__=="__main__":
    with open("cache_desc.yaml") as fin:
        datadict = yaml.load(fin)

    datadict = datadict["cachedict"]
    cache_nodes_dict = {}
    rank_id_dict = {}
    cache_nodes_distmatrix = []
    for i, key in enumerate(datadict):
        rank = key
        rank_id_dict[i] = rank

        ip = datadict[key][0].split(":")[0]
        port = datadict[key][0].split(":")[1]
        cache_nodes_dict[i] = [ip, port]
    
    num_of_rank = len(rank_id_dict)
    for i in range(num_of_rank):
        distdict = datadict[rank_id_dict[i]][1]
        cache_nodes_distmatrix.append([])

        for j in range(num_of_rank):
            distdict_key = ":".join(cache_nodes_dict[j])
            cache_nodes_distmatrix[i].append(distdict[distdict_key])

    chain = generate_mindistchain(cache_nodes_distmatrix)

    print(cache_nodes_dict)
    print(rank_id_dict)
    print(cache_nodes_distmatrix)
    print(chain)