# Author: baichen318@gmail.com


import os
import time
import yaml
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, Optional, NoReturn, Union


def parse_args() -> argparse.Namespace:
    def initialize_parser(
        parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        parser.add_argument(
        	"-c",
            "--configs",
            required=True,
            type=str,
            default="configs.yml",
            help="YAML file to be handled"
        )

        return parser

    parser = argparse.ArgumentParser(
    	formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser = initialize_parser(parser)
    return parser.parse_args()


def get_configs(fyaml: str) -> Dict:
    if_exist(fyaml, strict=True)
    with open(fyaml, 'r') as f:
        try:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        except AttributeError:
            configs = yaml.load(f)
    return configs


def if_exist(path: str, strict: bool = False) -> Optional[bool]:
	if os.path.exists(path):
		return True
	else:
		warn("{} is not found.".format(path))
		if strict:
			exit(1)
		else:
			return False


def mkdir(path: str) -> NoReturn:
    if not if_exist(path):
        info("create directory: {}.".format(path))
        os.makedirs(path, exist_ok=True)


def remove(path: str) -> NoReturn:
    if if_exist(path):
        if os.path.isfile(path):
            os.remove(path)
            info("remove {}.".format(path))
        elif os.path.isdir(path):
            if not os.listdir(path):
                # empty directory
                os.rmdir(path)
            else:
                shutil.rmtree(path)
            info("remove {}.".format(path))


def load_excel(path: str, sheet_name: Union[str, int] = 0) -> pd.core.frame.DataFrame:
    """
        path: excel path
    """
    if_exist(path, strict=True)
    data = pd.read_excel(path, sheet_name=sheet_name)
    info("read the sheet {} of excel from {}.".format(sheet_name, path))
    return data


def write_txt(path: str, data: np.ndarray, fmt: str = "%i") -> NoReturn:
    """
        `path`: path to the output path
        `data`: <np.array>
    """
    dims = len(data.shape)
    if dims > 2:
        warn("cannot save the shape {} to {}.".format(
            data.shape, path)
        )
        return
    info("save the ndarray to {}.".format(path))
    np.savetxt(path, data, fmt)


def execute(cmd: str, logger=None):
    if logger:
        logger.info("executing: {}.".format(cmd))
    else:
        info("executing: {}".format(cmd))
    return os.system(cmd)


def execute_with_subprocess(cmd, logger=None):
    if logger:
        logger.info("executing: {}.".format(cmd))
    else:
        print("[INFO]: executing: {}.".format(cmd))
    subprocess.call(["bash", "-c", cmd])


def timestamp() -> float:
    return time.time()


def remove_suffix(s: str, suffix: str) -> str:
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    else:
        return s[:]


def info(msg: str) -> NoReturn:
    print("[INFO]: {}".format(msg))


def test(msg: str) -> NoReturn:
    print("[TEST]: {}".format(msg))


def warn(msg: str) -> NoReturn:
    print("[WARN]: {}".format(msg))


def error(msg: str) -> NoReturn:
    print("[ERROR]: {}".format(msg))
    exit(1)


def assert_error(msg: str) -> str:
	return "[ERROR]: {}".format(msg)
