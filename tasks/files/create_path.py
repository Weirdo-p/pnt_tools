
import os
import sys

# n = sys.argv[1]

if len(sys.argv) != 2: 
    print("ERROR: please input a number")
    exit(-1)
n = int(sys.argv[1])
for i in range(1, n + 1):
    file_name = 'HC%02d' % i
    if (os.path.exists(file_name)):continue
    os.mkdir(file_name)

print("create successfully")