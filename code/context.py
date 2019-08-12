import os
import sys


thisdir = os.path.dirname(__file__)
rootdir = os.path.join(thisdir, '.')

if rootdir not in sys.path:
    sys.path.insert(0, rootdir)
    os.chdir(rootdir)
