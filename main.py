# Author: baichen318@gmail.com


import os
import sys
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "algo")
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "utils")
)
from utils import get_configs, parse_args
from algo.boom_explorer import boom_explorer


def main():
    boom_explorer(configs)


if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    main()
