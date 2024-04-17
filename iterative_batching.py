from logging import currentframe
from networkx import shortest_path_length
from networkx.algorithms.connectivity import build_auxiliary_node_connectivity
from numpy import who
import copy
import pandas as pd
import itertools

from offline import parse_all_pair_shortest_paths

from utils import depicklify

'''returns true if we can add a order to a existing_batch'''
def is_it_batch(existing_batch, new_order, shortest_path):
    existing_batch.append(new_order)
    cust_nodes = [idx for idx in range(len(existing_batch))]
    all_pickedup_ts = max(order.prep_ts for order in existing_batch)

    permutations = itertools.permutations(cust_nodes)
    
    def get_perm(permutations):
        for permutation in (permutations):
            curr_node = new_order.rest_node
            curr_time = all_pickedup_ts
            curr_dist = 0
            flag = True
            for order in permutation:
                node = existing_batch[order].cust_node
                time_requirement = existing_batch[order].deliver_ts
                try:
                    time_required = shortest_path[curr_node][node]['time']
                    dist_required = shortest_path[curr_node][node]['dist']
                except:
                    time_required = shortest_path[node][curr_node]['time']
                    dist_required = shortest_path[node][curr_node]['dist']
                
                if(curr_time + time_required <= time_requirement):
                    curr_time += time_required
                    curr_node = copy.deepcopy(node)
                    curr_dist += dist_required
                else:
                    flag = False
                    break
            if flag:
                return (True, permutation, curr_dist)
        return (False, None, 0)

    (is_possible, ideal_perm, curr_dist) = get_perm(permutations)
    if is_possible and ideal_perm is not None: 
        res_batch = []
        for idx in ideal_perm:
            res_batch.append(existing_batch[idx])
        return (True, res_batch, curr_dist)
    else:
        return (False, None, 0)

def compress_batch(batch, curr_deliver_dist):
    rest_node = batch[0].rest_node
    cust_node = batch[-1].cust_node
    placed_ts = max(order.placed_ts for order in batch)
    prep_ts = max(order.prep_ts for order in batch)
    deliver_ts = max(order.deliver_ts for order in batch)
    prep_time = prep_ts - placed_ts
    deliver_time = deliver_ts - prep_ts
    # deliver_dist = max(order.deliver_ts for order in batch)
    deliver_dist = curr_deliver_dist
    return [rest_node, cust_node, placed_ts, prep_ts, deliver_ts, prep_time, deliver_time, deliver_dist]


def convert_batches_to_dataframe(batches, dists):
    cols = ["rest_node", "cust_node", "placed_ts", "prep_ts", "deliver_ts", "prep_time", "deliver_time", 
                    "deliver_dist"]
    res = []
    # for rests in batches:
    for rests_idx in range(len(batches)):
        rests = batches[rests_idx]
        for ind_idx in range(len(rests)):
            ind_batch = rests[ind_idx]
            curr_deliver_dist  = dists[rests_idx][ind_idx]
            if type(curr_deliver_dist) == list:
                curr_deliver_dist = curr_deliver_dist[0]

            res.append(compress_batch(ind_batch, curr_deliver_dist))

    df= pd.DataFrame(res)
    df.columns = cols
    df.to_csv('batching_utils/temp.csv')
    return df

'''Takes a list or dataframe of orders and then tries to batch them into clusters of size atmost batch_size'''
def batching(orders, batch_size, shortest_path, num_nodes):
    batches = []
    dists = []
    for node_idx in range(num_nodes):
        batches.append([])
        dists.append([])
    

    for idx, order in orders.iterrows():
        if len(batches[order.rest_node]) == 0:
            batches[order.rest_node].append([order])
            dists[order.rest_node].append([order.deliver_dist])
        else:
            flag = False
            for batch_idx in range(len(batches[order.rest_node]) ):

                '''check whether adding order to the already pre-existing order makes a viable batch'''
                batch = batches[order.rest_node][batch_idx]
                (is_possible, ideal_perm, curr_dist) = is_it_batch(batch, order, shortest_path)
                if(is_possible):
                    flag = True
                    batches[order.rest_node][batch_idx] = ideal_perm
                    dists[order.rest_node][batch_idx] = curr_dist
                    break
            if flag == False:
                batches[order.rest_node].append([order])
                dists[order.rest_node].append([order.deliver_dist])
                
    return convert_batches_to_dataframe(batches, dists)



# orders_data = pd.read_csv('data/X/orders/orders_250_500_1000.csv')
# apsp_dist = depicklify('data/X/map/apsp_dist_500_p0.1.pkl')
# apsp_time = depicklify('data/X/map/apsp_time_500_p0.1.pkl')
# apsp_ = parse_all_pair_shortest_paths(apsp_dist, apsp_time)
# batches = batching(orders_data, 3, apsp_, 500)
# print(convert_batches_to_dataframe(batches))
