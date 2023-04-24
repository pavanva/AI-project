#cleans project model folder of the first param

from utils import data_paths
import sys

import os

PATHS = data_paths(sys.argv[1], create_if_missing=False)

if os.path.exists(PATHS['last']):
    os.remove(PATHS['last'])

if os.path.exists(PATHS['log']):
    os.remove(PATHS['log'])

if os.path.exists(PATHS['best']):
    os.remove(PATHS['best'])