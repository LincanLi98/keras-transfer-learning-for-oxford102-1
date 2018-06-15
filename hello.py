import os
import os.path
from os.path import join

x = sorted(o for o in os.listdir("g:/") if os.path.isdir("g:/"+o))

print(x)