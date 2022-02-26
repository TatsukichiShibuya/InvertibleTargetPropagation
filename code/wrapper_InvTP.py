import os
import sys
args = ' '.join(map(str, sys.argv[1:]))
command = f'python Main.py --algorithm=InvTP {args} --type=C --log --agent'
print(command)
os.system(command)
