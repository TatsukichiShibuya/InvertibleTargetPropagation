import os
import sys
args = ' '.join(map(str, sys.argv[1:]))
command = f'python Main.py --algorithm=DTTP --problem=classification --datasize=70000 --epochs=1000 --depth=4 --direct_depth=2 {args}'
print(command)
os.system(command)
