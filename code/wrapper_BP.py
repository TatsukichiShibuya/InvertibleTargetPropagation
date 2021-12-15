import os
import sys
args = ' '.join(map(str, sys.argv[1:]))
command = f'python Main.py --algorithm=BP {args} --log'
print(command)
os.system(command)
