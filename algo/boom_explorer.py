# Author: baichen318@gmail.com


from tqdm import tqdm
from typing import NoReturn
from solver import create_solver
from problem import create_problem


def boom_explorer(configs: dict) -> NoReturn:
	problem = create_problem(configs)
	solver = create_solver(problem)

	solver.initialize()
	iterator = tqdm(range(configs["bo"]["max-bo-steps"]))
	for step in iterator:
		iterator.set_description("Iter {}".format(step + 1))
		solver.fit_dkl_gp()
		solver.eipv_suggest()
	solver.report()
