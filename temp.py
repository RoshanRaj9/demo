from itertools import permutations

# Define the list
my_list = ['i1', 'i2', 'i3', 'r1', 'r2', 'r3']

# Generate all permutations
all_permutations = permutations(my_list)

# Filter permutations where i1 comes before r1, i2 comes before r2, and i3 comes before r3
filtered_permutations = [
    perm for perm in all_permutations
    if perm.index('i1') < perm.index('r1') and
       perm.index('i2') < perm.index('r2') and
       perm.index('i3') < perm.index('r3')
]

# Print the filtered permutations
for perm in filtered_permutations:
    print(perm)

