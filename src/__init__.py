import sys
from os.path import dirname, abspath, join

cur_folder = dirname(abspath(__file__))
sys.path.insert(0, join(dirname(cur_folder), 'src'))