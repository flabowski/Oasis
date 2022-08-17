#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:45:42 2022

@author: florianma
"""
import dolfin as df
import numpy as np
from oasis.common import parse_command_line
from oasis.problems import info_red, OasisTimer, initial_memory_use, oasis_memory
from oasis.problems.NSfracStep.Cylinder import Cylinder as ProblemDomain
import oasis.solvers.NSfracStep.IPCS_ABCN as solver
from low_rank_model_construction.basis_function_interpolation import ReducedOrderModel
import matplotlib.pyplot as plt
from shutil import copy2
import matplotlib
import json

config_file = "../config.json"
config = json.load(open(file=config_file, encoding="utf-8"))


np.set_printoptions(suppress=True)
matplotlib.use("Agg")  # Qt5Agg for normal interactive, Agg for offsceen
plt.close("all")

commandline_args = parse_command_line()
default_problem = "Cylinder"
problemname = config.get("problem", default_problem)
print("Importing problem module " + problemname)
# TODO: import the right ProblemDomain based on problemname:
# ProblemDomain =  NSfracStep.get_domain(problemname)

my_domain = ProblemDomain(config)
# my_domain.set_parameters(config)
my_domain.set_parameters(commandline_args)
my_domain.mesh_from_file(config["mesh_name"], config["facet_name"])
# df.plot(my_domain.mesh)
# my_domain.recommend_dt()
# Create lists of components solved for
my_domain.initialize_problem_components()
# Declare FunctionSpaces and arguments
my_domain.dolfin_variable_declaration()
# Boundary conditions
my_domain.create_bcs()
# TODO: Read in previous solution if restarting
# TODO: LES setup
# TODO: Non-Newtonian setup.
# Initialize solution
my_domain.apply_bcs()
solvername = my_domain.solver
# TODO: import the right solver based on solvername:
# solver =  NSfracStep.get_solver(solvername)
max_iter = config["max_iter"]
max_error = config["max_error"]
use_ROM = config["use_ROM"]
pressure_step = config["pressure_step"]
velocity_correction_step = config["velocity_correction_step"]
debug = config["debug"]
decompose_results = config["decompose_results"]

if use_ROM:
    dir_oasis = "/home/florianma@ad.ife.no/ad_disk/Florian/Repositoties/Oasis/"
    dir_rom = dir_oasis + "results/cylinder_reference/"

    U = np.load(dir_rom + "ROM_U_p.npy")
    S = np.load(dir_rom + "ROM_S_p.npy")
    VT = np.load(dir_rom + "ROM_VT_p.npy")
    X_min = np.load(dir_rom + "ROM_X_min_p.npy")
    X_range = np.load(dir_rom + "ROM_X_range_p.npy")
    X = np.load(dir_rom + "X_p.npy")
    time = np.load(dir_rom + "time.npy")
    nu = np.load(dir_rom + "nu.npy")
    grid = (time,)
    r = np.sum(np.cumsum(S) / np.sum(S) < 0.9999)  # -0.00373 ... 0.00450
    ROM = ReducedOrderModel(grid, U[:, :r], S[:r], VT[:r], X_min, X_range)

tx = OasisTimer("Timestep timer")
tx.start()
total_timer = OasisTimer("Start simulations", True)

cond = my_domain.krylov_solvers["monitor_convergence"]
print_info = my_domain.use_krylov_solvers and cond  # = print_solve_info
it0 = my_domain.iters_on_first_timestep
max_iter = my_domain.max_iter

fit = solver.FirstInner(my_domain)
tvs = solver.TentativeVelocityStep(my_domain)
ps = solver.PressureStep(my_domain)
stop = False
t = 0.0
tstep = 0
total_inner_iterations = 0
while (t - df.DOLFIN_EPS) < (my_domain.T - my_domain.dt) and not stop:

    # if tstep == 30:
    #     stop = True

    # t += my_domain.dt  # avoid annoying rounding errors
    t = np.round(t + my_domain.dt, decimals=8)
    tstep += 1

    inner_iter = 0
    num_iter = max(it0, max_iter) if tstep <= 10 else max_iter

    ts = OasisTimer("start_timestep_hook")
    my_domain.start_timestep_hook(t)  # update bcs
    ts.stop()

    tr0 = OasisTimer("ROM")
    if use_ROM:
        if tstep > 1:
            offset = 0.0  # -0.5 * my_domain.dt
        else:
            offset = 0.0
        guess1 = my_domain.q_["p"].vector().vec().array.copy()
        guess2 = ROM.predict([t - offset])[0].ravel()
        guess3 = X[:, tstep - 1]  # only works if dt is the same..
        my_domain.q_["p"].vector().vec().array = guess2.copy()
    tr0.stop()
    udiff = 1e8
    while udiff > max_error and inner_iter < num_iter:
        inner_iter += 1
        total_inner_iterations += 1

        t0 = OasisTimer("Tentative velocity")
        if inner_iter == 1:
            # lesmodel.les_update()
            # nnmodel.nn_update()
            fit.assemble()
            tvs.A = fit.A
        udiff = 0
        for i, ui in enumerate(my_domain.u_components):
            t1 = OasisTimer("Solving tentative velocity " + ui, print_info)
            tvs.assemble(ui=ui)  # uses p_ to compute gradp
            my_domain.velocity_tentative_hook(ui=ui)
            udiff = tvs.solve(ui=ui, udiff=udiff)
            t1.stop()
        t0.stop()

        t2 = OasisTimer("Pressure solve", print_info)
        if tstep % pressure_step == 0 or tstep <= 10:
            ps.assemble()
            my_domain.pressure_hook()
            pdiff = ps.solve()
        t2.stop()
        # discard the extra inner iterations of the first 10 outer itartions
        if inner_iter < max_iter:
            if hasattr(my_domain, "udiff"):
                my_domain.udiff[tstep - 1, inner_iter - 1] = udiff
            if hasattr(my_domain, "pdiff"):
                my_domain.pdiff[tstep - 1, inner_iter - 1] = pdiff

        if debug:
            p = my_domain.q_["p"].vector().vec().array
            e3 = np.abs(p - guess3)
            phi = np.abs(my_domain.dp_.vector().vec().array)
            print(
                inner_iter,
                "{:.6f}\t{:.8f}\t{:.8f}".format(udiff, phi.max(), e3.max()),
            )
        my_domain.print_velocity_pressure_info(num_iter, inner_iter, udiff)
    if debug:
        print(
            "step {:.0f}, time: {:.6f} s. Inner loop stopped after {:.0f}"
            " inner iterations. Total inner iterations: {:.0f}".format(
                tstep, t, inner_iter, total_inner_iterations
            )
        )
    # if use_ROM:
    #     p = my_domain.q_["p"].vector().vec().array
    #     e1 = np.abs(p - guess1)
    #     e2 = np.abs(p - guess2)
    #     e3 = np.abs(p - guess3)
    #     print(
    #         "{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t".format(
    #             e1.max(), e1.mean(), e2.max(), e2.mean(), e3.max(), e3.mean()
    #         )
    #     )
    # Update velocity
    t3 = OasisTimer("Velocity update")
    if tstep % velocity_correction_step == 0 or tstep <= 10:
        for i, ui in enumerate(my_domain.u_components):
            tvs.velocity_update(ui=ui)
    t3.stop()

    # TODO: Solve for scalars
    # if len(scalar_components) > 0:
    #     solver.scalar_assemble()
    #     for ci in scalar_components:
    #         t1 = OasisTimer("Solving scalar {}".format(ci), print_solve_info)
    #         pblm.scalar_hook()
    #         solver.scalar_solve()
    #         t1.stop()
    t4 = OasisTimer("temporal hook")
    my_domain.temporal_hook(t=t, tstep=tstep, ps=ps, tvs=tvs)
    t4.stop()

    # TODO: Save solution if required and check for killoasis file
    # stop = io.save_solution()
    my_domain.advance()

    # Print some information
    if tstep % my_domain.print_intermediate_info == 0:
        toc = tx.stop()
        my_domain.show_info(t, tstep, toc)
        df.list_timings(df.TimingClear.clear, [df.TimingType.wall])
        tx.start()

    # AB projection for pressure on next timestep
    if (
        my_domain.AB_projection_pressure
        and t < (my_domain.T - tstep * df.DOLFIN_EPS)
        and not stop
    ):
        my_domain.q_["p"].vector().axpy(0.5, my_domain.dp_.vector())

total_timer.stop()
df.list_timings(df.TimingClear.keep, [df.TimingType.wall])
info_red("Total computing time = {0:f}".format(total_timer.elapsed()[0]))
oasis_memory("Final memory use ")
# total_initial_dolfin_memory
m = df.MPI.sum(df.MPI.comm_world, initial_memory_use)
info_red("Memory use for importing dolfin = {} MB (RSS)".format(m))
info_red("Total memory use of solver = {:.4f} MB (RSS)".format(oasis_memory.memory - m))
# Final hook
my_domain.theend_hook(SVD=decompose_results)
# TODO: save data that was actually used. DO that in the end hook
pth = my_domain.pkg_dir + "results/" + my_domain.simulation_start + "/"
copy2(config_file, pth + config_file.split("/")[-1])
