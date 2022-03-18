#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:01:22 2022

@author: florianma
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from os import listdir, remove, rename, mkdir, fsdecode
from os.path import isfile, join, dirname, exists
from datetime import datetime
from oasis.problems.NSfracStep import FracDomain
from dolfin import Expression, DirichletBC, Mesh, XDMFFile, MeshValueCollection
from dolfin import cpp
from ROM.snapshot_manager import Data, load_snapshots_cavity, load_snapshots_cylinder
from low_rank_model_construction.proper_orthogonal_decomposition import row_svd

H = 0.41
L = 2.2
D = 0.1
center = 0.2
cases = {1: {"Um": 0.3, "Re": 20.0}, 2: {"Um": 1.5, "Re": 100.0}}


class Cylinder(FracDomain):
    # run for Um in [0.2, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0]
    # or  for Re in [20., 50., 60., 75.0, 100, 150, 200]
    def __init__(self, case=1):
        """
        Create the required function spaces, functions and boundary conditions
        for a channel flow problem
        """
        super().__init__()
        # problem parameters
        # case = parameters["case"] if "case" in parameters else 1
        Umax = cases[case]["Um"]  # 0.3 or 1.5
        # Re = cases[case]["Re"]  # 20 or 100
        Umean = 2.0 / 3.0 * Umax
        rho = 1.0
        mu = 0.001
        self.H = 0.41
        Re = rho * Umean * D / mu
        print("Re", Re)
        nu = mu / rho
        self.Umean = Umean
        self.Umax = Umax
        self.Schmidt = {}
        self.Schmidt_T = {}
        self.nu = nu
        self.T = 10
        self.dt = 0.01
        self.checkpoint = 50
        self.save_step = 50
        self.plot_interval = 10
        self.velocity_degree = 2
        self.print_intermediate_info = 100
        self.use_krylov_solvers = True
        # self.krylov_solvers["monitor_convergence"] = True
        self.velocity_krylov_solver["preconditioner_type"] = "jacobi"
        self.velocity_krylov_solver["solver_type"] = "bicgstab"
        self.NS_expressions = {}
        self.scalar_components = []
        self.pkg_dir = pkg_dir = dirname(__file__).split("oasis")[0]
        self.simulation_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.temp_dir = temp_dir = pkg_dir + "results/tmp/"
        msg = "is not empty. Do you want to remove all its content? [y/n]:"
        if exists(temp_dir):
            if len(listdir(temp_dir)) > 0:
                if input(temp_dir + msg) == "y":
                    for file in listdir(temp_dir):
                        filename = fsdecode(file)
                        remove(temp_dir + filename)
                else:
                    raise ValueError(temp_dir + "needs to be empty")
        else:
            mkdir(temp_dir)
        return

    def mesh_from_file(self, mesh_name, facet_name):
        self.mesh = Mesh()
        with XDMFFile(mesh_name) as infile:
            infile.read(self.mesh)
        dim = self.mesh.topology().dim()
        mvc = MeshValueCollection("size_t", self.mesh, dim - 1)
        with XDMFFile(facet_name) as infile:
            infile.read(mvc, "name_to_read")
        self.mf = cpp.mesh.MeshFunctionSizet(self.mesh, mvc)
        self.bc_dict = {
            "fluid": 0,
            "channel_walls": 1,
            "cylinderwall": 2,
            "inlet": 3,
            "outlet": 4,
        }
        return

    def create_bcs(self):
        mf, bc_dict = self.mf, self.bc_dict
        V, Q, Umax, H = self.V, self.Q, self.Umax, self.H
        # U0_str = "4.0*x[1]*(0.41-x[1])/0.1681"
        U0_str = "4.*{0}*x[1]*({1}-x[1])/pow({1}, 2)".format(Umax, H)
        inlet = Expression(U0_str, degree=2)
        bc00 = DirichletBC(V, inlet, mf, bc_dict["inlet"])
        bc01 = DirichletBC(V, 0.0, mf, bc_dict["inlet"])
        bc10 = bc11 = DirichletBC(V, 0.0, mf, bc_dict["cylinderwall"])
        bc2 = DirichletBC(V, 0.0, mf, bc_dict["channel_walls"])
        bcp = DirichletBC(Q, 0.0, mf, bc_dict["outlet"])
        self.bcs = {
            "u0": [bc00, bc10, bc2],
            "u1": [bc01, bc11, bc2],
            "p": [bcp],
        }
        return

    def start_timestep_hook(self, **kvargs):
        """Called at start of new timestep"""
        # TODO: predict p_
        # X_approx, X_approx_n = self.ROM.predict(xi)
        # self.q_["p"].vector().vec().array = X_approx.ravel()
        pass

    def temporal_hook(self, t, tstep, **kvargs):
        i = tstep - 1
        pth = self.temp_dir
        if i % 100 == 0:
            fig, (ax1, ax2) = self.plot()
            plt.savefig(pth + "frame_{:06d}.png".format(i))
            plt.close()
        # u = self.q_["u0"].compute_vertex_values(mesh)  # 2805
        u = self.q_["u0"].vector().vec().array  # 10942
        v = self.q_["u1"].vector().vec().array
        p = self.q_["p"].vector().vec().array
        np.save(pth + "u_{:06d}.npy".format(i), u)
        np.save(pth + "v_{:06d}.npy".format(i), v)
        np.save(pth + "p_{:06d}.npy".format(i), p)
        tvs = kvargs.get("tvs", None)
        if tvs:
            gradpx = tvs.gradp["u0"].vector().vec().array
            gradpy = tvs.gradp["u1"].vector().vec().array
            np.save(pth + "gradpx_{:06d}.npy".format(i), gradpx)
            np.save(pth + "gradpy_{:06d}.npy".format(i), gradpy)

        return

    def theend_hook(self):
        print("post processing:")
        pth = self.pkg_dir + "results/" + self.simulation_start + "/"
        mkdir(pth)
        tmp_pth = self.temp_dir
        # move temp png files to results folder
        for quantity in ["u", "v", "p", "gradpx", "gradpy"]:
            onlyfiles = [
                f
                for f in listdir(tmp_pth)
                if (isfile(join(tmp_pth, f)) and f.startswith(quantity + "_"))
            ]
            onlyfiles.sort()

            u = np.load(tmp_pth + onlyfiles[0])
            X_q = np.empty((len(u), len(onlyfiles)))
            for i, f in enumerate(onlyfiles):
                X_q[:, i] = np.load(tmp_pth + f)
            print(quantity, X_q.min(), X_q.max(), X_q.mean())
            np.save(pth + "X_" + quantity + ".npy", X_q)

            my_data = Data(X_q, False)
            X_n = my_data.normalise()
            U, S, VT = row_svd(X_n, 1, 1, False, False)
            np.save(pth + "ROM_U_" + quantity + ".npy", U)
            np.save(pth + "ROM_S_" + quantity + ".npy", S)
            np.save(pth + "ROM_VT_" + quantity + ".npy", VT)
            np.save(pth + "ROM_X_min_" + quantity + ".npy", my_data.X_min)
            np.save(pth + "ROM_X_range_" + quantity + ".npy", my_data.X_range)

            fig, ax = plt.subplots()
            plt.imshow(X_q)
            plt.show()
            for i, f in enumerate(onlyfiles):
                remove(tmp_pth + f)
        # move temp png files to results folder
        for f in listdir(tmp_pth):
            if isfile(join(tmp_pth, f)) and f.endswith(".png"):
                rename(tmp_pth + f, pth + f)
        self.dt
        # save meshes as well
        V, Q = self.V, self.Q
        np.save(pth + "V_dof_coords.npy", V.tabulate_dof_coordinates())
        np.save(pth + "Q_dof_coords.npy", Q.tabulate_dof_coordinates())
        np.save(pth + "mesh_coords.npy", V.mesh().coordinates())
        np.save(pth + "mesh_cells.npy", V.mesh().cells())
        t = np.arange(0.0, self.T, self.dt) + self.dt
        print("found", len(t), "timesteps and", len(onlyfiles), "saved files")
        np.save(pth + "time.npy", t)
        np.save(pth + "nu.npy", np.array([self.nu]))
        # np.save(pth + "Re.py", np.array([self.Re]))
        # TODO: save other parameters (mu, ...)

        print("finished :)")
        return

    def plot(self):
        # u, p = self.u_, self.p_
        mesh = self.mesh
        u = self.q_["u0"].compute_vertex_values(mesh)
        v = self.q_["u1"].compute_vertex_values(mesh)
        p = self.q_["p"].compute_vertex_values(mesh)
        # print(u.shape, v.shape, p.shape)
        magnitude = (u ** 2 + v ** 2) ** 0.5
        # print(u.shape, v.shape, p.shape, magnitude.shape)

        # velocity = u.compute_vertex_values(mesh)
        # velocity.shape = (2, -1)
        # magnitude = np.linalg.norm(velocity, axis=0)
        x, y = mesh.coordinates().T
        # u, v = velocity
        tri = mesh.cells()
        # pressure = p.compute_vertex_values(mesh)
        # print(x.shape, y.shape, u.shape, v.shape)
        fs = (12, 6)
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, figsize=fs)
        ax1.quiver(x, y, u, v, magnitude)
        ax2.tricontourf(x, y, tri, p, levels=40)
        ax1.set_aspect("equal")
        ax2.set_aspect("equal")
        ax1.set_title("velocity")
        ax2.set_title("pressure")
        return fig, (ax1, ax2)
