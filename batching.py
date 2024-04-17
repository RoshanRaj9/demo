from sys import exception
from networkx import tree_data
from numpy import append, identity
import pandas as pd
import itertools

from pandas.io.parquet import FastParquetImpl

from utils import *
from eval import *

class batch:
    def __init__(self, group, cost, weight, route):
        self.group = group 
        self.cost = cost
        self.weight = weight
        self.route = route

def write_apsp_time(apsp_, nodes):
    with open('batching_utils/apsp_time.txt', 'w') as file:
        for id1 in range(1,nodes+1):
            for id2 in range(1, nodes+1):
                try:
                    time = apsp_[id1][id2]['time']
                except:
                    time = apsp_[id2][id1]['time']
                file.write(str(time) + " ")
            file.write('\n')


def write_apsp_dist(apsp_, nodes):
    with open('batching_utils/apsp_dist.txt', 'w') as file:
        for id1 in range(1,nodes+1):
            for id2 in range(1, nodes+1):
                try:
                    dist = apsp_[id1][id2]['dist']
                except:
                    dist = apsp_[id2][id1]['dist']
                file.write(str(dist) + " ")
            file.write('\n')


def parse_all_pair_shortest_paths(apsp_dist, apsp_time):
    ''' 
    Merges the shortest path data calculated based on "time" and "dist" attributes.    '''
    print("Parsing all-pairs shortest path data ...") 
    outer_keys = list(apsp_dist.keys()) # rest_nodes active within t_hrs
    inner_keys = list(apsp_dist[outer_keys[0]].keys()) # all_nodes in road_network~40K

    print(f"Total number of nodes : {len(inner_keys)}")
    print(f"Number of restaurants active within the the first 't' hours : {len(outer_keys)}")

    all_pairs_shortest_paths = {
        node_1: {
            node_2: {'dist':0, 'time':0} for node_2 in inner_keys
        } for node_1 in outer_keys
    }
    
    for node_1 in tqdm(outer_keys, total=len(outer_keys)):
        for node_2 in inner_keys:
            all_pairs_shortest_paths[node_1][node_2]['dist'] = apsp_dist[node_1][node_2]
            all_pairs_shortest_paths[node_1][node_2]['time'] = apsp_time[node_1][node_2]
    
    return all_pairs_shortest_paths

def is_it_batch(existing_batch, new_order, shortest_path):
    for order in new_order:
        existing_batch.append(order)

    all_nodes = []
    for idx in range(len(existing_batch)):
        all_nodes.append("r" + str(idx))
        all_nodes.append("c" + str(idx))

    permutations = itertools.permutations(all_nodes)
    '''filtering permutations'''
    filtered_perms = []
    for perm in permutations:
        for idx in range(len(existing_batch)):
            if perm.index("r" + str(idx)) < perm.index("c" + str(idx)):
                filtered_perms.append(perm)

    
    permutations= filtered_perms
    def get_perm(permutations):
        for permutation in (permutations):
            curr_dist = 0
            flag = True

            curr_node = int(permutation[0][1:])
            curr_time = existing_batch[curr_node].prep_ts
            curr_node = existing_batch[curr_node].rest_node
            for node_idx in range(1, len(permutation)):
                type = permutation[node_idx][0]
                node = int(permutation[node_idx][1:])
                time_prep = existing_batch[node].prep_ts
                time_requirement = existing_batch[node].deliver_ts
                if(type == 'r'):
                    node = existing_batch[node].rest_node
                    '''pickup from restaurant node'''
                    try:
                        time_required = shortest_path[curr_node][node]['time']
                        dist_required = shortest_path[curr_node][node]['dist']
                    except:
                        time_required = shortest_path[node][curr_node]['time']
                        dist_required = shortest_path[node][curr_node]['dist']
                
                    if(curr_time + time_required <= time_requirement):
                        curr_time += time_required
                        curr_time = max(curr_time+time_required, time_prep)
                        curr_node = copy.deepcopy(node)
                        curr_dist += dist_required
                    else:
                        flag = False
                        break
                else:
                    node = existing_batch[node].cust_node
                    '''deliver at node'''
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
        res_plan = []
        for locs in ideal_perm:
            type = locs[0]
            node = int(locs[1:])
            
            if(type == 'r'):
                res_plan.append(existing_batch[node].rest_node)
            else:
                res_plan.append(existing_batch[node].cust_node)
            
        return (True, existing_batch, res_plan, curr_dist)
    else:
        return (False, None, None, 0)

def avg_weight(pi):
    sum = 0
    for batch in pi:
        sum += batch.weight
    sum = sum/len(pi)
    return sum


def batching(orders_data, eta, MAXO):
    '''line-1'''
    r = 0

    '''line-2'''
    pi = []
    for index, row in orders_data.iterrows():
        rest_node = row['rest_node']
        cust_node = row['cust_node']
        try:
            curr_cost = apsp_[rest_node][cust_node]['dist']
        except:
            curr_cost = apsp_[cust_node][rest_node]['dist']
        curr_order = batch( [row], curr_cost, 0, [rest_node, cust_node])
        pi.append(curr_order)

    '''line-3'''
    w = []
    for i in range(0, len(pi)-1):
        row1 = pi[i].group
        for j in range(i+1, len(pi)):
            row2 = pi[j].group
            (flag, res_batch, res_route, curr_dist) = is_it_batch( row1[:], row2[:], apsp_)
            if(flag == True):
                new_weight = curr_dist - (pi[i].cost + pi[j].cost)
                # print(curr_dist , "---", new_weight)
                curr_weight = batch(res_batch, curr_dist, new_weight, res_route) 
                w.append(curr_weight)
    w = sorted(w, key=lambda x:x.weight)

    '''line-4'''
    r = 0
    while True:
        condition = avg_weight(pi)
        print("Hello", r, condition)
        if len(pi) == 0 or condition > eta:
            break

        sigma_m = w[0]
        w = w[1:]
        if(len(sigma_m.group) > MAXO):
            continue
    
        '''removing pi_i and pi_j'''
        print("Before ", len(pi))
        new_pi = []
        for x in pi:
            flag = False
            for rhs in x.group:
                for lhs in sigma_m.group:
                    if rhs.equals(lhs):
                        flag = True
                        break
            if flag == False:
                new_pi.append(x)

        pi = new_pi
        print(len(pi))

        '''Inserting pi_i U pi_j'''
        pi.append(sigma_m)
        
        '''removing all edges incident to pi_i and pi_j'''
        new_wi = []
        for w_i in w:
            flag = False
            for rhs in w_i.group:
                for lhs in sigma_m.group:
                    if lhs.equals(rhs):
                        flag = True
                        break
            if flag == False:
                new_wi.append(w_i)
        w = new_wi


        '''Adding new edges using pi_i U pi_j'''
        print("Last step")
        order2 = sigma_m.group
        for i in range(0, len(pi)-1):
            order1 = pi[i].group
            if( len(order1) + len(order2) > MAXO):
                continue
            (flag, res_batch, res_route, curr_dist) = is_it_batch(order2, order1, apsp_)
            if flag == True:
                new_weight = curr_dist - (pi[i].cost + sigma_m.cost)
                print("curr_dist ", curr_dist, new_weight)
                curr_batch = batch(res_batch, curr_dist, new_weight, res_route)
                w.append(curr_batch)

        r += 1

    print("Pi size", len(pi))
    return pi


# Load the CSV file
orders_data = pd.read_csv('data/X/orders/orders_250_500_1000.csv')
apsp_dist = depicklify('data/X/map/apsp_dist_500_p0.1.pkl')
apsp_time = depicklify('data/X/map/apsp_time_500_p0.1.pkl')
apsp_ = parse_all_pair_shortest_paths(apsp_dist, apsp_time)

# print(orders_data)

nodes = 500
# write_apsp_time(apsp_, nodes)
# write_apsp_dist(apsp_, nodes)

'''Sort orders_data according to placed_ts values'''
orders_data = orders_data.sort_values(by='placed_ts')
(batching(orders_data, 2, 3))
