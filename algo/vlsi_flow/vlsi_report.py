# Author: baichen318@gmail.com


import os
import re
import numpy as np
from datetime import datetime
from typing import List, Tuple
from utils import if_exist, mkdir, write_txt


class Macros(object):
	def __init__(self, soc):
		super(Macros, self).__init__()
		self._soc = soc
		self.macros = {}
		self.macros["chipyard-research-root"] = os.path.abspath(
			os.path.join(
				os.path.dirname(__file__),
				os.path.pardir,
				os.path.pardir
			)
		)
		self.macros["vlsi-root"] = os.path.join(
			os.path.join(
				self.macros["chipyard-research-root"],
				"vlsi"
			)
		)

	@property
	def soc(self):
		return self._soc

	def get_syn_root(self):
		return os.path.join(
			self.macros["vlsi-root"],
			"build",
			"chipyard.TestHarness.{}-ChipTop".format(self.soc),
			"syn-rundir"
		)

	def get_bmark_sim_root(self, bmark):
		return os.path.join(
			self.macros["vlsi-root"],
			"build",
			"chipyard.TestHarness.{}-ChipTop".format(self.soc),
			"sim-syn-rundir",
			bmark
		)

	def get_bmark_sim_log(self, bmark):
		return os.path.join(
			self.get_bmark_sim_root(bmark),
			"{}.log".format(bmark)
		)

	def get_bmark_power_rpt(self, bmark):
		return os.path.join(
			self.get_bmark_sim_root(bmark),
			"power",
			"reports",
			"{}.power.avg.max.report".format(bmark)
		)

	def get_area_rpt(self):
		return os.path.join(
			self.get_syn_root(),
			"reports",
			"final_area.rpt"
		)


def generate_microarchitecture_embedding(design_space, idx):
	microarchitecture_embedding = \
		design_space.idx_to_vec(idx)
	return np.array(microarchitecture_embedding)


def generate_performance(configs, idx):
	def parse_sim_log(sim_log):
		instructions, cycles, ipc = 0, 0, 0
		with open(sim_log, 'r') as f:
			for line in f.readlines():
				if "[INFO]" in line and "cycles" in line and "instructions" in line:
					try:
						instructions = re.search(r'\d+\ instructions', line).group()
						instructions = int(instructions.split("instructions")[0])
						cycles = re.search(r'\d+\ cycles', line).group()
						cycles = int(cycles.split("cycles")[0])
						ipc = (instructions / cycles)
					except AttributeError:
						continue
		return instructions, cycles, ipc

	performance = []
	for bmark in configs["benchmarks"]:
		macros = Macros("Boom{}Config".format(idx))
		sim_log = macros.get_bmark_sim_log(bmark)
		if if_exist(sim_log):
			instructions, cycles, ipc = parse_sim_log(sim_log)
		else:
			instructions, cycles, ipc = 0, 0, 0
		performance.append(instructions)
		performance.append(cycles)
		performance.append(ipc)
	return np.array(performance)


def generate_power(configs, idx):
	def parse_power_rpt(power_rpt):
		power = 0
		with open(power_rpt, 'r') as f:
			for line in f.readlines():
				# NOTICE: extract the total power of BoomTile
				if "boom_tile (BoomTile)" in line:
					power = float(line.split()[-2])
		return power

	power = []
	for bmark in configs["benchmarks"]:
		macros = Macros("Boom{}Config".format(idx))
		power_rpt = macros.get_bmark_power_rpt(bmark)
		if if_exist(power_rpt):
			_power = parse_power_rpt(power_rpt)
		else:
			_power = 0
		power.append(_power)
	return np.array(power)


def generate_area(idx):
	def parse_area_rpt(area_rpt):
		area = 0
		with open(area_rpt, 'r') as f:
			for line in f.readlines():
				if "BoomTile" in line:
					area = float(line.split()[-1])
		return area

	area = []
	macros = Macros("Boom{}Config".format(idx))
	area_rpt = macros.get_area_rpt()
	if if_exist(area_rpt):
		_area = parse_area_rpt(area_rpt)
	else:
		_area = 0
	area.append(_area)
	return np.array(area)


def generate_dataset_impl(design_space: object, idx: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	dataset = []
	for _idx in idx:
		_dataset = np.array([])
		# generate microarchitecture embedding
		_dataset = np.concatenate(
			(_dataset, generate_microarchitecture_embedding(design_space, _idx))
		)
		# generate IPC
		_dataset = np.concatenate((_dataset, generate_performance(design_space.configs["vlsi"], _idx)))
		# generate power
		_dataset = np.concatenate((_dataset, generate_power(design_space.configs["vlsi"], _idx)))
		# generate area
		_dataset = np.concatenate((_dataset, generate_area(_idx)))
		# generate time	
		dataset.append(_dataset)
	dataset = np.array(dataset)
	mkdir(os.path.dirname(design_space.configs["vlsi"]["report"]))
	write_txt("{}-{}".format(
			configs["vlsi"]["report"], datetime.now().isoformat()
		),
		dataset,
		fmt="%f"
	)
	return dataset[:,-3], dataset[:,-2], dataset[:,-1]


def get_report(design_space: object, idx: List[int]) -> Tuple[float, float, float]:
	return generate_dataset_impl(design_space, idx)
