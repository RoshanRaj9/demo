import argparse
import os
import pandas as pd
from utils import *
from eval import *
from iterative_batching import *
import ast

'''Gurobi Linear Optimizer'''
import gurobipy as gb
from gurobipy import GRB

'''Logging'''
import gc 
import logging

def get_shape(lst):
    if not isinstance(lst, list):
        return []
    return [len(lst)] + get_shape(lst[0])

def interval_in_sorted_list(intervals, target_interval, start, end):
    if(start > end):
        return False

    mid = (start+end)//2 
    if(intervals[mid][0] <= target_interval[0] and 
            intervals[mid][1] >= target_interval[1]):
        return True 
    
    if(intervals[mid][0] > target_interval[0]):
        return interval_in_sorted_list(intervals, target_interval, start, mid-1)
    else:
        if(intervals[mid][1] < target_interval[1]):
            return interval_in_sorted_list(intervals, target_interval, mid+1, end)
        else:
            return True

def parse_orders(requests_data):
    ''' 
    Converts the orders data into the required "node" format of the flow network.
    '''
    print("Parsing requests data ...")
    # Store v_{br_i}'s, v_{er_i}'s, and v_{tr_i}'s for all requests r_i; i \in [NUM_REQUESTS]
    begin_nodes = []    # (rest_node, placed_timestamp) 
    end_nodes = []      # (rest_node, food_prep_timestamp)
    terminal_nodes = [] # (cust_node, deliver_timestamp) 
    requests = []       # (v_br, v_er, v_tr)

    for _idx, request in tqdm(requests_data.iterrows(), total=requests_data.shape[0]):
        # gather data
        v_br = (request.rest_node, request.placed_ts)
        v_er = (request.rest_node, request.prep_ts)
        v_tr = (request.cust_node, request.deliver_ts)
        request_ds = (v_br, v_er, v_tr) 
        # store data
        begin_nodes.append(v_br) 
        end_nodes.append(v_er)
        terminal_nodes.append(v_tr)
        requests.append(request_ds)

    return begin_nodes, end_nodes, terminal_nodes, requests

def parse_servers(servers_data):
    '''
    This will parse the servers_data and returns us 2 useful results 
    first is the servers_data which has times for where each server is active 
    second is a dictionary which will gives us what servers are active at a particular time
    '''
    servers_data['available_timestamps'] = servers_data['available_timestamps'].apply(ast.literal_eval)
    timeline_dict = {}
    active_time = [0 for _ in range(NUM_SERVERS)]
    for timestamp in range(1, NUM_TIMESTEPS+1):
        timeline_dict[timestamp] = []

    for idx, server in servers_data.iterrows():
        for timelines in server.available_timestamps:
            left_timeline = timelines[0]
            right_timeline = timelines[1]
            active_time[idx] += right_timeline - left_timeline + 1
            for times in range(left_timeline, right_timeline+1):
                timeline_dict[times].append(idx)

    return servers_data, timeline_dict, active_time

def parse_all_pair_shortest_paths(apsp_dist, apsp_time):
    ''' 
    Merges the shortest path data calculated based on "time" and "dist" attributes.
    '''
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

def construct_and_set_variables(model, begin_nodes, end_nodes, terminal_nodes, all_pairs_shortest_paths, weight_var,  servers_data):
    ''' 
        We maintain separate flow-variables for each server.
        "timestamps" in {1,2,...,NUM_TIMESTAMPS}
    '''

    num_total_vars = 0
    all_in_out_vars = []
    all_sink_vars = []
    all_source_vars = []
    all_end_terminal_vars = []
    all_end_terminal_costs = []
    all_into_end_vars = []
    all_into_end_costs = []
    infeasibility_vars = []
    min_reward_var = None


    '''Type-1,2,3,4,5 Variables'''
    for server_idx in tqdm(range(NUM_SERVERS), total=NUM_SERVERS):
        in_out_vars = {} 
        into_end_vars = []
        into_end_costs = []
        end_terminal_vars = []
        end_terminal_costs = []
        print("num_timesteps", NUM_TIMESTAMPS)
        for node_idx in (ALL_NODES):
            for timestamp in range(1, NUM_TIMESTAMPS+1, ODD):
                node = (node_idx, timestamp) 
                in_out_vars[node] = {'in':[], 'out':[]}


        '''Type-1 : Source-node Variables'''
        '''shape of all_source_vars -> (NUM_SERVERS x TIMELINES x NUM_NODES)'''
        print("Adding Type-1 Variables")
        source_vars = []
        for timelines in servers_data.loc[server_idx, 'available_timestamps']:
            starting_time = timelines[0]
            source_var_timeline = []
            for node_idx in (ALL_NODES):
                src_var = model.addVar(lb = 0.0, ub = 1.0, 
                                       vtype = GRB.CONTINUOUS,
                                       name = f"{server_idx}_source_to_{node_idx}_at_{starting_time}")
                source_var_timeline.append(src_var)
                num_total_vars += 1
                in_out_vars[(node_idx, starting_time)]['in'].append(src_var)
            source_vars.append(source_var_timeline)
        all_source_vars.append(source_vars)
        

        '''Type-2 Node-sink variables'''
        print("Adding Type-2 Variables")
        '''shape of all_sink_vars -> (NUM_SERVERS x TIMELINES x NUM_NODES)'''
        sink_vars = []
        for timelines in servers_data.loc[server_idx, 'available_timestamps']:
            ending_time = min(timelines[1], NUM_TIMESTAMPS)
            sink_var_timeline = []
            for node_idx in (ALL_NODES):
                sink_var = model.addVar(lb = 0.0, ub = 1.0, 
                                       vtype = GRB.CONTINUOUS,
                                       name = f"{server_idx}_sink_to_{node_idx}_at_{ending_time}")
                sink_var_timeline.append(sink_var)
                num_total_vars += 1
                in_out_vars[(node_idx, ending_time)]['out'].append(sink_var)
            sink_vars.append(sink_var_timeline)
        all_sink_vars.append(sink_vars)


        '''Type-3 Self Variables'''
        '''No need to store these self_vars in a dictionary'''
        print("Adding Type-3 Variables")
        for timelines in servers_data.loc[server_idx,'available_timestamps']:
            starting_time = timelines[0]
            ending_time = min(timelines[1], NUM_TIMESTAMPS)
            for times in range(starting_time, ending_time):
                for node_idx in (ALL_NODES):
                    self_var = model.addVar(lb = 0.0, ub = 1.0, 
                                       vtype = GRB.CONTINUOUS,
                                    name=f"{server_idx}_self_{node_idx}_{times}_to_{times+1}")
                    in_out_vars[(node_idx, times)]['out'].append(self_var)
                    in_out_vars[(node_idx, times+1)]['in'].append(self_var)
                    num_total_vars += 1

        '''Type-4 Request-time variables'''
        ''' Shape of into_end_vars -> NUM_SERVERS x NUM_REQUESTS x NUM_NODES , be careful that some of 
            them can be None as well'''
        print("Adding Type-4 Variables")
        is_server_eligible = []
        for request_idx in range(NUM_REQUESTS):
            is_server_eligible.append(0)
        for request_idx in range(NUM_REQUESTS):
            begin_node = begin_nodes[request_idx]
            end_node = end_nodes[request_idx]
            terminal_node = terminal_nodes[request_idx]
            
            assert(begin_node[0] == end_node[0])

            placed_timestamp = begin_node[1]
            prepared_timestamp = end_node[1]
            prep_time = prepared_timestamp - placed_timestamp

            curr_request_vars = []
            curr_request_costs = []

            for node_idx in (ALL_NODES):
                try:
                    shortest_path_time = all_pairs_shortest_paths[node_idx][end_node[0]]['time']
                except:
                    shortest_path_time = all_pairs_shortest_paths[end_node[0]][node_idx]['time']

                if(node_idx == end_node[0]):
                    '''already covered by self_vars'''
                    continue
                
                curr_node_var = None
                curr_node_cost = None

                prepared_timestamp = int(prepared_timestamp)
                placed_timestamp = int(placed_timestamp)
                for timestamp in range(prepared_timestamp-1, placed_timestamp, -1):
                    intervals = servers_data['available_timestamps'][server_idx]
                    flag = interval_in_sorted_list( intervals, [timestamp, terminal_node[1]], 0, len(intervals)-1)
                    if(flag == False):
                        '''Current server cannot deliver this order'''
                        continue
                    
                    is_server_eligible[request_idx] = 1
                    # candidate_end_node = (node_idx, timestamp)
                    if(shortest_path_time <= prep_time):
                        ien_var = model.addVar(
                                lb=0.0, ub=1.0, 
                                vtype=GRB.CONTINUOUS, 
                                name=f"{server_idx}_request-{request_idx}:from-{(node_idx,timestamp)}_to_end-{end_node}"
                                )

                        if weight_var=='dist':
                            # scaling_factor = 15 # ??
                            try:
                                cost = all_pairs_shortest_paths[node_idx][end_node[0]]['dist'] 
                            except:
                                cost = all_pairs_shortest_paths[end_node[0]][node_idx]['dist']
                            # cost += (prepared_timestamp-timestamp)*scaling_factor # in order to account for waiting time!
                        else:
                            scaling_factor = 10 # ??
                            cost = shortest_path_time + scaling_factor*(prepared_timestamp - timestamp) # waiting time

                        in_out_vars[(node_idx, timestamp)]['out'].append(ien_var)
                        in_out_vars[end_node]['in'].append(ien_var)
                        curr_node_var = ien_var
                        curr_node_cost = cost
                        num_total_vars += 1
                        break


                if curr_node_var is not None:
                    curr_request_vars.append(curr_node_var)
                    curr_request_costs.append(curr_node_cost)
                # curr_request_vars.append(curr_node_var)
                # curr_request_costs.append(curr_node_cost)

            assert len(curr_request_vars) <= NUM_NODES #cuz we only add one variable per node
            
            into_end_vars.append(curr_request_vars)
            into_end_costs.append(curr_request_costs)


        '''Type-5 End-Terminal Variables'''
        ''' Shape of end_terminal_vars -> NUM_SERVERS x NUM_REQUESTS, be careful that some of them can be 
            None as well '''
        print("Adding type 5 Variables")
        for request_idx in range(NUM_REQUESTS):
            end_node = end_nodes[request_idx]
            terminal_node = terminal_nodes[request_idx]

            if(weight_var == 'time'):
                prepared_timestamp= end_node[1]
                delivered_timestamp = terminal_node[1]
                cost = int(delivered_timestamp - prepared_timestamp)
            else:
                cost = int(orders_data.iloc[request_idx].deliver_dist)

            end_terminal_costs.append(cost)
            if( is_server_eligible[request_idx] == 1):
                et_var = model.addVar( lb= 0.0, ub=1.0,  vtype = GRB.CONTINUOUS,
                                        name=f"{server_idx}_end-{end_node}_to_terminal-{terminal_node}")
                end_terminal_vars.append(et_var)
                in_out_vars[end_node]['out'].append(et_var)
                in_out_vars[terminal_node]['in'].append(et_var)
                num_total_vars += 1
            else:
                end_terminal_vars.append(None)

        all_in_out_vars.append(in_out_vars)
        all_into_end_vars.append(into_end_vars)
        all_into_end_costs.append(into_end_costs)
        all_end_terminal_vars.append(end_terminal_vars)
        all_end_terminal_costs.append(end_terminal_costs)

        ''' since it can occupy GBs of RAM and it's going to be reinitialized in the next iteration! So this memory needs to be freed! '''
        del in_out_vars         
        gc.collect()


    '''Type-6 Infeasibility Variables'''
    print("Adding Type-6 Variables")
    for request_idx in range(NUM_REQUESTS):
        infeasibility_var = model.addVar(
                                         
                                        vtype=GRB.BINARY,
                                        name=f"unserved_{request_idx}"
                                        )
        infeasibility_vars.append(infeasibility_var)
        num_total_vars += 1

    '''Type-7 Minimum Reward Variable'''
    print("Adding Type-7 Variables")
    min_reward_var = model.addVar(lb=0.0,  vtype=GRB.CONTINUOUS, 
                                  name="minimum reward/earning accumulated by any server") # NOT doing obj=1.0 here,
    num_total_vars += 1

    all_vars_and_costs = {
            "in_out_vars" : all_in_out_vars,
            "source_vars" : all_source_vars,
            "sink_vars" : all_sink_vars,
            "end_terminal_vars" : all_end_terminal_vars,
            "end_terminal_costs" : all_end_terminal_costs,
            "into_end_vars" : all_into_end_vars,
            "into_end_costs" : all_into_end_costs,
            "infeasibility_vars" : infeasibility_vars,
            "min_reward_var" : min_reward_var
        }

    print(f"TOTAL NUMBER OF *VARIABLES* IN THE MODEL : {num_total_vars}")
    logger.info(f"TOTAL NUMBER OF *VARIABLES* IN THE MODEL : {num_total_vars}")

    return all_vars_and_costs

def construct_and_set_constraints(model, vars_and_costs, only_last_mile):
    num_total_constrs = 0
    (   
        in_out_vars,
        source_vars,
        sink_vars,
        end_terminal_vars,
        end_terminal_costs,
        into_end_vars,
        into_end_costs,
        infeasibility_vars,
        min_reward_var
    ) = vars_and_costs.values()
    server_rewards = []

    first_mile_costs = into_end_costs
    last_mile_costs = end_terminal_costs
    

    '''Type-1 Source_nodes flow Constraints'''
    for server_idx in range(NUM_SERVERS):
        for source_groups in source_vars[server_idx]:
            model.addConstr(gb.quicksum(source_groups[node_idx] for node_idx in range(NUM_NODES) ) == 1)
            num_total_constrs += 1

    '''Type-2 Sink_nodes flow Constraints'''
    for server_idx in range(NUM_SERVERS):
        for sink_groups in sink_vars[server_idx]:
            model.addConstr( gb.quicksum( sink_groups[node_idx] for node_idx in range(NUM_NODES) ) == 1)
            num_total_constrs += 1

    '''Type-3 Flow conservation '''
    for server_idx in range(NUM_SERVERS):
        in_out_data = in_out_vars[server_idx]
        for timelines in servers_data.loc[server_idx, 'available_timestamps']:
            left_timeline = timelines[0]
            right_timeline = min(timelines[1], NUM_TIMESTAMPS)
            for times in range(left_timeline, right_timeline+1):
                for node_idx in (ALL_NODES):
                    curr_node = (node_idx, times)
                    in_nodes = in_out_data[curr_node]['in']
                    out_nodes = in_out_data[curr_node]['out']
                    model.addConstr(gb.quicksum(in_nodes)==gb.quicksum(out_nodes))
                    num_total_constrs += 1
    

    '''Type-4 Minimum Reward Constraint '''
    for server_idx in range(NUM_SERVERS):
        first_mile_vars = into_end_vars[server_idx]
        last_mile_vars = end_terminal_vars[server_idx]

        first_mile_reward = gb.LinExpr()

        for request_idx in range(NUM_REQUESTS):
            curr_fm_vars = first_mile_vars[request_idx]
            curr_fm_costs = first_mile_costs[server_idx][request_idx]
            for flow_var, reward_value in zip(curr_fm_vars, curr_fm_costs):
                # if flow_var is not None:
                first_mile_reward.addTerms(reward_value, flow_var)

        # last_mile_reward = gb.quicksum((flow_var * reward_value) for flow_var, reward_value in zip(last_mile_vars, last_mile_costs[server_idx]))
        last_mile_reward = gb.LinExpr()
        for flow_var, reward_value in zip(last_mile_vars, last_mile_costs[server_idx]):
            if flow_var is not None:
                last_mile_reward.addTerms(reward_value, flow_var)

        if(only_last_mile): curr_server_reward = last_mile_reward
        else: curr_server_reward = last_mile_reward + first_mile_reward

        if(curr_server_reward.size() == 0):
            continue
        else:
            curr_server_reward = curr_server_reward/active_time[server_idx]
            server_rewards.append(curr_server_reward)

            model.addConstr(min_reward_var <= curr_server_reward)
            num_total_constrs += 1

    '''Type-5 Net input flow into end node should be <= 1'''
    first_mile_vars = into_end_vars
    for req_idx in range(NUM_REQUESTS):
        tot_vars = gb.LinExpr() 
        for idx in range(NUM_SERVERS):
            curr_fm_vars = first_mile_vars[idx][req_idx]
            for var in curr_fm_vars:
                # if var is not None:
                tot_vars.addTerms(1, var) 
        model.addConstr(tot_vars, GRB.LESS_EQUAL, 1)

    '''Type-6 infeasibility_vars '''
    for request_idx in range(NUM_REQUESTS):
        curr_const_terms = []
        for server_idx in range(NUM_SERVERS):
            last_mile_vars = end_terminal_vars[server_idx][request_idx]
            if last_mile_vars is not None:
                curr_const_terms.append(last_mile_vars)

        curr_const_terms.append(1.0 * infeasibility_vars[request_idx])

        # assert len(curr_const_terms)==NUM_SERVERS+1
        model.addConstr(gb.quicksum(curr_const_terms) == 1)
        num_total_constrs += 1

    vars_and_costs['server_rewards'] = server_rewards
    print("Constraints added!")
    print(f"TOTAL NUMBER OF *CONSTRAINTS* IN THE MODEL : {num_total_constrs}")
    logger.info(f"TOTAL NUMBER OF *CONSTRAINTS* IN THE MODEL : {num_total_constrs}")
    return vars_and_costs

def k_server_general(begin_nodes, end_nodes, terminal_nodes, 
                     all_pairs_shortest_paths, weight_var, only_last_mile, objective_type, servers_data):
    ''' 
    Efficient and Feasible solution:
        for Efficiency: minimize the sum of ther server rewards 
        for Feasibility: either set a very high infeasibility penalty 
                        or add constraints that all infeasibility variables should be 0. (for this, make sure that there are enough number of servers) 
    '''
    ''' Step-1: Instantiate gurobi Model '''
    print("INSTANTIATING FLOW-BASED MIP MODEL ...")
    model_name = "flow_mip"
    model = gb.Model("Flow-network based MIP model")
    # model.setParam('NonConvex', 2)

    ''' Step-2: Add Variables '''
    print("ADDING VARIABLES ...")
    vars_and_costs = construct_and_set_variables(model, begin_nodes, end_nodes, terminal_nodes, all_pairs_shortest_paths, weight_var, servers_data) 

    '''Pruning with "dist" as cost is fine (equivalent to case with no pruning) 
                                            BUT pruning should not be done with "time" as cost.'''
    assert weight_var=='dist'


    ''' Step-3: Add Constraints '''
    print("ADDING CONSTRAINTS ...")
    vars_and_costs = construct_and_set_constraints(model, vars_and_costs, only_last_mile) 
    min_reward_var = vars_and_costs['min_reward_var']
    server_rewards = vars_and_costs['server_rewards']
    infeasibility_vars = vars_and_costs['infeasibility_vars']
    if weight_var=='dist':
        infeasibility_penalty_1 = (MAX_DELIVERY_DIST/NUM_SERVERS) 
        infeasibility_penalty_2 = (MEAN_DELIVERY_DIST/NUM_SERVERS)
        infeasibility_penalty = 1e20 * MAX_DELIVERY_DIST # universal
        # infeasibility_penalty = 0
    # elif weight_var=='time':
    else:
        infeasibility_penalty_1 = (MAX_DELIVERY_TIME/NUM_SERVERS) 
        infeasibility_penalty_2 = (MEAN_DELIVERY_TIME/NUM_SERVERS)
        infeasibility_penalty = 1e20 * MAX_DELIVERY_TIME # universal

    def get_total_cost_lower_bound(weight_var, only_last_mile):
        ''' 
        Note that min_total_cost is not actually the minimum total cost for given instance of the problem (for that you'd have to solve the mip for the 'min' objective)
        rather it's a lower bound on the total_cost across all possible feasible instances; feasible=>availability of sufficient number of servers (i.e., assuming no infeasibility).  
        '''
        min_total_cost = 0
        max_total_cost = 0
        total_last_mile_cost = 0
        total_first_mile_cost = 0
        
        into_end_node_costs = vars_and_costs['into_end_costs'][1]
        min_filtered_costs = [] 
        max_filetered_costs = []

        for request_costs in into_end_node_costs:
            flattened_request_costs = [] 
            for x in request_costs:
                if x is not None:
                    flattened_request_costs.append(x)
            try:
                curr_min_edge_cost = min(flattened_request_costs)
            except:
                curr_min_edge_cost = 0
            try:
                curr_max_edge_cost = max(flattened_request_costs)
            except:
                curr_max_edge_cost = 0 
            min_filtered_costs.append(curr_min_edge_cost) 
            max_filetered_costs.append(curr_max_edge_cost)

        min_first_mile_cost = sum(min_filtered_costs) # tight lower bound
        max_first_mile_cost = sum(max_filetered_costs) # loose lower bound

        if weight_var=='dist':
            total_last_mile_cost = np.sum(orders_data.deliver_dist.values) 
        elif weight_var=='time':
            total_last_mile_cost = np.sum(orders_data.deliver_time.values)

        if only_last_mile:
            min_total_cost = total_last_mile_cost 
            max_total_cost = total_last_mile_cost
        else:
            min_total_cost = total_last_mile_cost + min_first_mile_cost
            max_total_cost = total_last_mile_cost + max_first_mile_cost
        return min_total_cost, max_total_cost 


    min_total_cost, max_total_cost = get_total_cost_lower_bound(weight_var, only_last_mile)
    print(f"MINIMUM POSSIBLE COST: {min_total_cost}")
    print(f"MAXIMUM POSSIBLE COST: {max_total_cost}")
    logger.info(f"MINIMUM POSSIBLE COST: {min_total_cost}")
    logger.info(f"MAXIMUM POSSIBLE COST: {max_total_cost}")


    objectives = {}
    objectives['maxmin'] = (min_reward_var - infeasibility_penalty * gb.quicksum(z for z in infeasibility_vars))*(-1.0) # multiplied by -1.0 => MINIMIZATION 
    # objectives['maxmin'] = gb.quicksum(z for z in infeasibility_vars) # multiplied by -1.0 => MINIMIZATION 
    objectives['min'] = gb.quicksum(server_rewards) + infeasibility_penalty * gb.quicksum(z for z in infeasibility_vars) 
    # objectives['multi'] = gb.quicksum(server_rewards) - (NUM_SERVERS * min_reward_var) + infeasibility_penalty * gb.quicksum(z for z in infeasibility_vars)
    objectives['multi'] = 100 * gb.quicksum(server_rewards) - (NUM_SERVERS * min_reward_var) + 100 * gb.quicksum(z for z in infeasibility_vars)
    # in 'multi' objective, we try to minimize the net server_rewards AND maximize the min_server_reward
    
    if objective_type=='bound':
        # for 'bound', we use the maxmin objective only but we'll add one more constraint;
        # the idea is that the sum of maxmin server_rewards should be bounded by a factor of the 'min' possible sum(server_rewards)
        alpha = bound_factor
        # model.addConstr(gb.quicksum(server_rewards)*(1)<=alpha*min_total_cost)
        objective = objectives['maxmin']
    else:
        objective = objectives[objective_type]    

    # the 'multi' objective is *not* really a strict 'bound' objective 
    model.setObjective(objective, sense=GRB.MINIMIZE)

    ## Step-5: Solve 
    print("SOLVING ...")
    solved_model = optimize_model(model, model_name)
    
    return solved_model, vars_and_costs

def optimize_model(model, model_name):
    model.Params.Threads = 16
    # model.Params.NodefileStart = 0.5
    model.optimize()
    status = model.status 
    assert status==GRB.OPTIMAL, "An optimal solution could not be found !"
    if city!='X':
        path = os.path.join(data_path, 'results/offline', f'{model_name}_{NUM_REQUESTS}_{NUM_SERVERS}_{weight_var}.sol')
    else:
        path = os.path.join(data_path, 'results/offline', f'{model_name}_{NUM_REQUESTS}_{NUM_SERVERS}_{NUM_NODES}_{weight_var}.sol')
    # model.write()
    return model

def get_server_rewards(vars_and_costs, NUM_SERVERS, NUM_REQUESTS, ALL_NODES, only_last_mile):
    (   
        in_out_vars,
        source_vars,
        sink_vars,
        end_terminal_vars,
        end_terminal_costs,
        into_end_vars,
        into_end_costs,
        infeasibility_vars,
        min_reward_var, 
        server_rewards
    ) = vars_and_costs.values()

    final_rewards = []
    for server_idx in range(NUM_SERVERS):
        first_mile_vars = into_end_vars[server_idx]
        last_mile_vars = end_terminal_vars[server_idx]

        first_mile_reward = 0.0

        for request_idx in range(NUM_REQUESTS):
            curr_fm_vars = first_mile_vars[request_idx]
            curr_fm_costs = into_end_costs[server_idx][request_idx]
            for flow_var, reward_value in zip(curr_fm_vars, curr_fm_costs):
                if flow_var is not None:
                    first_mile_reward += reward_value*flow_var.X

        last_mile_reward = 0.0
        for flow_var, reward_value in zip(last_mile_vars, end_terminal_costs[server_idx]):
            if flow_var is not None:
                last_mile_reward += flow_var.X * reward_value

        if(only_last_mile): curr_server_reward = last_mile_reward
        else: curr_server_reward = last_mile_reward + first_mile_reward

        # curr_server_reward = curr_server_reward/active_time[server_idx]
        final_rewards.append(curr_server_reward)
    
    return final_rewards


if __name__=='__main__':
    # get program inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', choices=['X'], default='X', 
                        type=str, required=False, help='City name')
    parser.add_argument('--num_timesteps', choices=[100,200,500,1000,2000],
                        type=int, required=True)
    parser.add_argument('--num_nodes', choices=[50,100,500,1000], 
                        type=int, required=True, help='# nodes in metric space')
    parser.add_argument('--num_requests', choices=[10,30,50,100,250,500,1000],
                        type=int, required=True)
    parser.add_argument('--num_servers', choices=[5,10,20,25,30,50,100,150,200,250],
                        type=int, required=True)
    parser.add_argument('--edge_prob', choices=[0.1,0.2,0.5,0.6,0.7,0.9], default=0.5, 
                        type=float, required=False)
    parser.add_argument('--weight_var', choices=['time', 'dist'], default='dist',
                        type=str, required=False, help='Weight variable for road network')
    parser.add_argument('--only_last_mile', choices=[0, 1], default=False, 
                        type=int, required=True, help='Consider only last-mile rewards (or costs)')
    parser.add_argument('--objective', choices=['maxmin', 'min', 'multi', 'bound'], default='maxmin', 
                        type=str, required=False, help='Type of Objective function')
    parser.add_argument('--bound_factor', choices=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 5.0, 10.0], default=2.0,
                        type=float, required=False, help='optimal solution\'s of "bound" objective must be within "bound_factor" times the minimum possible cost')
    parser.add_argument('--ub', choices=[0, 1],
                        type=int, required=True, help='upper bound on the flow of end_terminal_edges; 0=>Symmetry Optimization employed!')
    parser.add_argument('--init', choices=[0,1], default=0,
                        type=int, required=False, help='Consider the initial locations of the servers')
    parser.add_argument('--batch_size', choices=[1,2,3,4,5,6,7,8,9,10], default=1, 
                        type=int, required=False, help='Average batch-/cluster-size when "batching" is done')
    parser.add_argument('--batching_type', choices=[1,2,3], default = 1,
                        type=int, required=False, help='Which batching technique have to be used')
    args = parser.parse_args()

    # create input variables
    city = args.city   
    NUM_TIMESTEPS = args.num_timesteps
    NUM_NODES = args.num_nodes
    NUM_REQUESTS = args.num_requests 
    NUM_SERVERS = args.num_servers
    edge_prob = args.edge_prob
    weight_var = args.weight_var
    only_last_mile = args.only_last_mile
    objective = args.objective
    bound_factor = args.bound_factor
    UB = args.ub
    INIT = args.init
    batching_type = args.batching_type
    
    ODD = 1

    data_path = '/Users/chey/10thsem/kfood/data'
    logs_path = os.path.join(data_path, city, 'logs')
    ints_path = os.path.join(data_path, city, 'drivers/de_intervals')
    drivers_init_path = os.path.join(data_path, f'{city}/drivers/init_nodes_{NUM_REQUESTS}_{NUM_NODES}.csv')

    print("Loading data ...") 
    orders_data = pd.read_csv(os.path.join(data_path, 
                                    f'{city}/orders/orders_{NUM_REQUESTS}_{NUM_NODES}_{NUM_TIMESTEPS}.csv'))
    apsp_dist = depicklify(os.path.join(data_path, f'{city}/map/apsp_dist_{NUM_NODES}_p{edge_prob}.pkl'))
    apsp_time = depicklify(os.path.join(data_path, f'{city}/map/apsp_time_{NUM_NODES}_p{edge_prob}.pkl'))
    road_net = pd.read_csv(os.path.join(data_path, f'{city}/map/metric_space_{NUM_NODES}_p{edge_prob}.csv')) 
    servers_data = pd.read_csv("servers.csv")
    all_pairs_shortest_paths = parse_all_pair_shortest_paths(apsp_dist, apsp_time)

    ALL_NODES = range(1, NUM_NODES+1)

    if batching_type == 2:
        orders_data = batching(orders_data, 3, all_pairs_shortest_paths, NUM_NODES)
        NUM_REQUESTS = len(orders_data)
    elif batching_type == 3:
        orders_data = pd.read_csv('~/10thsem/kfood/batching_utils/batched_orders.csv')
        NUM_REQUESTS = len(orders_data)

    print(orders_data)
    #MODEL INPUTS 
    begin_nodes, end_nodes, terminal_nodes, requests = parse_orders(orders_data)
    servers_data, timeline_dict, active_time = parse_servers(servers_data)


    max_deliver_timestamp = orders_data['deliver_ts'].max()
    _NUM_SERVERS = min(NUM_SERVERS, NUM_REQUESTS)
    NUM_TIMESTAMPS = int(max_deliver_timestamp)  # time-step = 1 second # 1 day = 86400 seconds
    # NUM_TIMESTAMPS = NUM_TIMESTEPS

    ALL_TIMESTAMPS = np.arange(NUM_TIMESTAMPS) 
    NUM_VARS = NUM_NODES * NUM_TIMESTAMPS # final number of nodes in the LP and Flow network
    MAX_FPT = np.max(orders_data.prep_time.values)
    MEAN_DELIVERY_TIME = np.mean(orders_data.deliver_time.values)
    MEAN_DELIVERY_DIST = np.mean(np.array(orders_data.deliver_dist.values))
    MAX_DELIVERY_TIME = np.max(orders_data.deliver_time.values)
    MAX_DELIVERY_DIST = np.max(orders_data.deliver_dist.values)

    ###### INPUT SUMMARY BEGINS ###########
    print(f"# Points in the metric space (or # Nodes in road network): {NUM_NODES}")
    print(f"# Total time stamps possible: {NUM_TIMESTAMPS}")
    print(f"Number of requests (or orders): {NUM_REQUESTS}")
    print(f"Number of servers (or drivers): {_NUM_SERVERS}")

    global logger
    log_filename = ''
    if INIT: log_filename = f"INIT_{NUM_REQUESTS}_{_NUM_SERVERS}_{NUM_NODES}_{weight_var}.log" 
    else: log_filename = f"{NUM_REQUESTS}_{_NUM_SERVERS}_{NUM_NODES}_{weight_var}.log"
    logger = get_logger(logs_path, log_filename) 

    logger.info(f"NUM_NODES : {NUM_NODES}")
    logger.info(f"NUM_REQUESTS : {NUM_REQUESTS}")
    logger.info(f"NUM_SERVERS : {_NUM_SERVERS}")
    logger.info(f"NUM_TIMESTAMPS : {NUM_TIMESTAMPS}")
    ########## INPUT SUMMARY ENDS ################# 
    # breakpoint()
    # SOLVING #
    solve_start_time = time.time()  
    print("Optimization started ...")
    model, vars_and_costs = k_server_general(begin_nodes, end_nodes, terminal_nodes, 
                                            all_pairs_shortest_paths, weight_var, only_last_mile, objective, servers_data)
    solve_end_time = time.time() 
    solve_time = solve_end_time - solve_start_time
    print(f"Execution time : {solve_time/3600} hrs")
    logger.info(f"Execution time : {solve_time/3600} hrs")
    # breakpoint()

    # Evaluation:
    num_inf, request_feasibilities = calculate_infeasibility(vars_and_costs)
    unserved_percentage = (num_inf*100)/NUM_REQUESTS    
    print(f"{num_inf} out of {NUM_REQUESTS} requests ({unserved_percentage}%) remain unserved!")
    logger.info(f"{num_inf} out of {NUM_REQUESTS} requests ({unserved_percentage}%) remain unserved!")
    logger.info(f"Request feasibilities : {request_feasibilities}")


    print()
    server_rewards = get_server_rewards(vars_and_costs, NUM_SERVERS, NUM_REQUESTS, ALL_NODES, only_last_mile)
    # if server_rewards==1 : server_rewards[0] is the sum of the server_rewards of all _NUM_SERVERS 
    for s_idx, reward in enumerate(server_rewards):
        print(f"Reward of server #{s_idx}: {reward}")
        logger.info(f"Reward of server #{s_idx}: {reward}")

    optimal_cost = get_optimal_cost(model) 
    print(f"Optimal cost : {optimal_cost}")
    logger.info(f"Optimal cost : {optimal_cost}")

    gini = gini_index(server_rewards)
    print(f"Gini index: {gini}")
    logger.info(f"Gini index: {gini}")

    avg_dist = get_avg_distance(server_rewards)
    print(f"Avg. Dist. : {avg_dist}")
    logger.info(f"Avg. Dist. : {avg_dist}")
# '''
