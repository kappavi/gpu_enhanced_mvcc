#!/usr/bin/env python3

# used this to generate the very large csv file. 
# can modify this later for making more large file s
with open('test_very_large.csv', 'w') as f:
    for i in range(100_000):
        obj_id = i % 1000
        value = i * 10
        f.write(f"{obj_id},{value}\n")

print("generated test_very_large.csv with 100,000 rows") 