import csv
import copy
import random
import sys

num_timestamps = int(sys.argv[1])
num_servers = int(sys.argv[2])

data = [ ["server_id", "available_timestamps"] ]
data = []

for_0 = [[1, num_timestamps]]
# for timestamp in range(1, num_timestamps):
#     for_0.append(timestamp)
#
# for_others = []
# for timestamp in range(1, num_timestamps+1):
#     for_others.append(timestamp)


for server_id in range(0, num_servers):
    data.append( [server_id, for_0] )


# Write data to CSV file
with open('servers.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['server_id', 'available_timestamps'])
    for row in data:
        writer.writerow(row)
    
print("servers.csv file generated")
