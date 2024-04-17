#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include<iostream>
// #include<bits/stdc++.h>
#include <istream>
#include <iterator>
#include <sstream>
#include <string>
#include <strings.h>
#include <tuple>
#include <vector>

using namespace std;

struct order{
    double rest_node;
    double cust_node;
    double placed_ts;
    double prep_ts;
    double deliver_ts;
    double prep_time;
    double deliver_time;
    double deliver_dist;
};

bool operator==(const order& lhs, const order& rhs) {
    return lhs.cust_node == rhs.cust_node && 
        lhs.rest_node == rhs.rest_node &&
        lhs.prep_ts == rhs.prep_ts && 
        lhs.deliver_ts == rhs.deliver_ts &&
        lhs.deliver_dist == rhs.deliver_dist &&
        lhs.deliver_time == rhs.deliver_time &&
        lhs.prep_ts == rhs.prep_ts &&
        lhs.prep_time == rhs.prep_time ;
}


struct batch{
    vector<order> group;
    double cost;
    double weight;
    vector<string> route;
};

bool compareByplaced(order a, order b) {
    return a.placed_ts < b.placed_ts;
}

bool cmp_by_wt(batch a, batch b){
    return a.weight < b.weight;
}

vector<vector<double>> read_apsp(string type){
    ifstream file("batching_utils/apsp_" + type + ".txt");
    int nodes = 501;

    //Reading apsp_time file..
    vector<vector<double>> apsp_data;
    if(file.is_open()){
        string line;
        while( getline(file, line)){
            stringstream ss(line);
            string token;
            vector<double> tokens;

            // Parse the line into tokens separated by spaces
            while (getline(ss, token, ' ')) {
                tokens.push_back(stod(token));
            }
            apsp_data.push_back(tokens);
            if(tokens.size() != 500)cout << "broken\n";
        }

    }
    return apsp_data;
}

vector<order> read_csv(string orders_path){
    ifstream file(orders_path);
    vector<order> orders_data;

    string line;
    getline(file, line);
    while(getline(file, line)){
        stringstream ss(line);
        string cell;
        order curr;
        
        //rest_node
        getline(ss, cell, ',');
        curr.rest_node = stod(cell);

        //cust_node
        getline(ss, cell, ',');
        curr.cust_node = stod(cell);

        //placed_ts
        getline(ss, cell, ',');
        curr.placed_ts = stod(cell);

        //prep_ts
        getline(ss, cell, ',');
        curr.prep_ts = stod(cell);

        //deliver_ts
        getline(ss, cell, ',');
        curr.deliver_ts = stod(cell);

        //prep_time
        getline(ss, cell, ',');
        curr.prep_time = stod(cell);
        
        //deliver_time
        getline(ss, cell, ',');
        curr.deliver_time = stod(cell);

        //deliver_dist
        getline(ss, cell, ',');
        curr.deliver_dist = stod(cell);
        orders_data.push_back(curr);
    }
    return orders_data;
}

vector<vector<string>> generate_perms(vector<string> a, int len){
    vector<vector<string>> res;
    
    do{
        bool flag = true;
        for(int idx = 0; idx < len; idx++){
            int index1 = find(a.begin(), a.end(), "r" + to_string(idx)) -a.begin();
            int index2 = find(a.begin(), a.end(), "c" + to_string(idx)) -a.begin();
            if(index1 > index2)flag = false;
        }
        if(flag)res.push_back(a);
    }while(next_permutation(a.begin(), a.end()));

    return res;

}

double avg_weight(vector<batch> pi){
    double sum = 0;
    for(int i= 0; i< pi.size(); i++)sum += pi[i].weight;
    sum = sum/pi.size();
    return sum;
}

tuple<bool, vector<string>, double> is_it_batch(vector<order> existing_batch, vector<order> new_order, 
        vector<vector<double>> apsp_dist, vector<vector<double>> apsp_time){
    // existing_batch.push_back(new_order);
    for(auto order: new_order) existing_batch.push_back(order);

    vector<string> all_nodes;
    for(int idx = 0; idx < existing_batch.size(); idx++){
        all_nodes.push_back("r" + to_string(idx));
        all_nodes.push_back("c" + to_string(idx));
    }

    vector<vector<string>> perms = generate_perms(all_nodes, existing_batch.size());
    // for(auto i: perms){
    //     for(auto j: i)cout << j << " ";
    //     cout << endl;
    // }
    for(auto perm: perms){
        double curr_node = stod(perm[0].substr(1, perm[0].size()-1));
        curr_node = existing_batch[curr_node].rest_node;
        double curr_time = existing_batch[curr_node].prep_ts;
        bool flag = true;
        double curr_dist = 0;
        
        for(int idx=1; idx< perm.size(); idx++){
            char type = perm[idx][0];
            double node = stod(perm[idx].substr(1, perm[idx].size()-1));
            double time_requirement = existing_batch[node].deliver_ts;
            if(type == 'r'){
                node = existing_batch[node].rest_node;
                double time_required = apsp_time[curr_node][node];
                double dist_required = apsp_dist[curr_node][node];
                
                if(curr_time + time_required <= time_requirement){
                    curr_time = max(curr_time + time_required, existing_batch[node].prep_ts);
                    curr_node = node;
                    curr_dist += dist_required;
                }
                else{
                    flag = false;
                    break;
                }
            }
            else{
                node = existing_batch[node].cust_node;
                double time_required = apsp_time[curr_node][node];
                double dist_required = apsp_dist[curr_node][node];
                if(curr_time + time_required <= time_requirement){
                    curr_time += time_required;
                    curr_node = node;
                    curr_dist += dist_required;
                }
                else{
                    flag = false;
                    break;
                }
            }
        }
        if(flag){
            // vector<double> route_plan;
            // for(int idx =0; idx < perm.size(); idx++){
            //     char type = perm[idx][0];
            //     int index = stoi(perm[idx].substr(1, perm[idx].size()-1));
            //     if(type == 'r') route_plan.push_back(existing_batch[index].rest_node);
            //     else route_plan.push_back(existing_batch[index].cust_node);
            // }
            return make_tuple(true, perm, curr_dist);
        }
    }
    vector<string> emptyvec;
    return make_tuple(false, emptyvec, 0.0);
}

vector<batch> batching(vector<order> orders_data, double eta, double MAXO, vector<vector<double>> apsp_dist, vector<vector<double>> apsp_time){

    // '''line-2'''
    vector<batch> pi;
    for(int idx = 0; idx < orders_data.size(); idx++){
        double curr_rest = orders_data[idx].rest_node;
        double curr_cust = orders_data[idx].cust_node;

        batch curr_order;
        curr_order.group = {orders_data[idx]};
        curr_order.cost = apsp_dist[curr_rest][curr_cust];
        curr_order.weight = 0.0;
        // curr_order.route = {curr_rest, curr_cust};
        curr_order.route = {"r0", "c0"};
        pi.push_back(curr_order);
    }


    // '''line-3'''
    vector<batch> w;
    for(int i = 0; i < pi.size()-1; i++){
        for(int j= i+1; j< pi.size(); j++){
            batch b1 = pi[i];
            batch b2 = pi[j];
            // auto res = is_it_batch( b1.group, b2.group[0], apsp_dist, apsp_time);
            bool flag; vector<string> route_plan; double cost; 
            tie(flag, route_plan, cost) = is_it_batch( b1.group, b2.group, apsp_dist, apsp_time);
            if(flag){
                // cout << "route_plan ";
                // for(auto x: route_plan)cout << x << " ";
                // cout << endl;
                double new_weight = cost - (b1.cost + b2.cost);
                batch new_batch;
                new_batch.group = {b1.group[0], b2.group[0]};
                new_batch.cost = cost;
                new_batch.weight = new_weight;
                new_batch.route = route_plan;
                w.push_back(new_batch);
            }
        }
    }
    sort(w.begin(), w.end(), cmp_by_wt);

    // '''line-4'''
    int r = 0;
    while(true){
        double condition = avg_weight(pi);
        cout << r << " " << condition << "\n";
        if(pi.size() == 0 or condition > eta){
            // cout << condition << "\n";
            break;
        }

        batch sigma_m = w[0];
        w.erase(w.begin());
        if( sigma_m.group.size() > MAXO)continue;

        // '''removing pi_i and pi_j'''
        vector<batch> new_pi;
        for(int idx = 0; idx < pi.size(); idx++){
            vector<order> curr = pi[idx].group;
            bool flag = true;
            for(int id = 0; id < curr.size(); id++){
                for(auto order: sigma_m.group){
                    if(order == curr[id]){
                        flag = false;
                        break;
                    }
                }
            }
            if(flag) new_pi.push_back(pi[idx]);
        }
        pi= new_pi;

        // '''Inserting pi_i U pi_j'''
        pi.push_back(sigma_m);
        
        // '''removing all edges incident to pi_i and pi_j'''
        vector<batch> new_w;
        for(int idx = 0; idx < w.size(); idx++){
            vector<order> curr = w[idx].group;
            bool flag = true;
            for(int id = 0; id < curr.size(); id++){
                for(auto order: sigma_m.group){
                    if(order == curr[id]){
                        flag = false;
                        break;
                    }
                }
            }
            if(flag) new_w.push_back(w[idx]);
        }
        w = new_w;
                

        // '''Adding new edges using pi_i U pi_j'''
        vector<order> order2 = sigma_m.group;
        for(int idx = 0; idx < pi.size()-1; idx++){
            vector<order> order1 = pi[idx].group;
            if(order1.size() + order2.size() > MAXO)continue;
            bool flag; vector<string> route_plan; double cost; 
            tie(flag, route_plan, cost) = is_it_batch( order2, order1, apsp_dist, apsp_time);
            if(flag){
                double new_weight = cost - (pi[idx].cost + sigma_m.cost);
                batch curr_batch;
                order2.insert(order2.end(), order1.begin(), order1.end());
                curr_batch.cost = cost;
                curr_batch.weight = new_weight;
                curr_batch.group = order2;
                curr_batch.route = route_plan;
                w.push_back(curr_batch);
            }
        }
        r += 1;
    }
    return pi;
}

int main (int argc, char *argv[]) {

    vector<vector<double>> apsp_time = read_apsp("time");
    vector<vector<double>> apsp_dist = read_apsp("dist");

    string orders_path = "data/X/orders/orders_250_500_1000.csv";
    vector<order> orders_data = read_csv(orders_path);

    //sorting the data according to arrival of the orders 
    sort(orders_data.begin(), orders_data.end(), compareByplaced);
    // for(int i= 0; i< orders_data.size(); i++){
    //     cout << "rest_node " << orders_data[i].rest_node << "\n";
    // }

    vector<batch> res = batching(orders_data, 3, 3, apsp_dist, apsp_time);

    cout << "res size " << res.size() << "\n";
    ofstream file("batching_utils/batched_orders.csv");

    if (!file.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return 0;
    }
    vector<string> headers = {"rest_node", 
                               "cust_node" , 
                               "placed_ts" , 
                               "prep_ts" , 
                               "deliver_ts" ,
                               "prep_time" , 
                               "deliver_time" , 
                               "deliver_dist"};
    for(int i= 0;i < headers.size(); i++){
        file << headers[i];
        if(i != headers.size()-1) file << ",";
    }
    file << "\n";
    int counter = 1;
    for(batch curr: res){
        string start_info = curr.route[0];
        string end_info = curr.route[curr.route.size()-1];

        double starting_node = stod(start_info.substr(1, start_info.size()-1));
        double ending_node = stod(end_info.substr(1, end_info.size()-1));

        // file << counter; file << ",";

        // string rest_node = to_string( static_cast<int>(curr.group[starting_node].rest_node) );
        auto rest_node = ( static_cast<int>(curr.group[starting_node].rest_node) );
        file << rest_node; file << ",";

        // string cust_node = to_string(static_cast<int>(curr.group[ending_node].cust_node) );
        auto cust_node = (static_cast<int>(curr.group[ending_node].cust_node) );
        file << cust_node; file << ",";

        // string placed_ts = to_string(static_cast<int>(curr.group[starting_node].placed_ts));
        auto placed_ts = (static_cast<int>(curr.group[starting_node].placed_ts));
        file << placed_ts; file << ",";

        // string prep_ts = to_string(static_cast<int>(curr.group[ending_node].prep_ts));
        auto prep_ts = (static_cast<int>(curr.group[ending_node].prep_ts));
        file << prep_ts; file << ",";

        // string deliver_ts = to_string(static_cast<int>(curr.group[ending_node].deliver_ts));
        auto deliver_ts = (static_cast<int>(curr.group[ending_node].deliver_ts));
        file << deliver_ts; file << ",";

        // string prep_time = to_string(static_cast<int>(curr.group[ending_node].prep_ts - curr.group[starting_node].placed_ts));
        auto prep_time = (static_cast<int>(curr.group[ending_node].prep_ts - curr.group[starting_node].placed_ts));
        file << prep_time; file << ",";

        // string deliver_time = to_string(static_cast<int>(curr.group[ending_node].deliver_ts - curr.group[ending_node].prep_ts));
        auto deliver_time = (static_cast<int>(curr.group[ending_node].deliver_ts - curr.group[ending_node].prep_ts));
        file << deliver_time; file << ",";

        auto deliver_dist = static_cast<int>(curr.cost);
        file << deliver_dist;

        file << "\n";
        counter += 1;
    }
    file.close();

    return 0;
}
