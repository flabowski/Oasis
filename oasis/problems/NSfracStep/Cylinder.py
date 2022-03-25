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
from shutil import copy2
from oasis.problems.NSfracStep import FracDomain
from dolfin import Expression, DirichletBC, Mesh, XDMFFile, MeshValueCollection
from dolfin import cpp, FacetNormal, grad, Identity, ds, assemble, dot
from ROM.snapshot_manager import Data
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
        self.coeff = 2 / (rho * D * Umean ** 2)
        self.Re = Re
        self.mu = mu
        self.rho = rho
        self.Umean = Umean
        self.Umax = Umax
        self.Schmidt = {}
        self.Schmidt_T = {}
        self.nu = nu
        self.T = 8
        self.dt = 1 / 1600
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
        m = int(self.T / self.dt)
        self.C_D = np.empty((m,))
        self.C_L = np.empty((m,))
        self.t_u = np.empty((m,))
        self.t_p = np.empty((m,))
        self.C_D[:] = self.C_L[:] = np.NaN
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
        self.ds_ = ds(subdomain_data=self.mf, domain=self.mesh)
        return

    def create_bcs(self):
        mf, bc_dict = self.mf, self.bc_dict
        V, Q, Umax, H = self.V, self.Q, self.Umax, self.H
        # U0_str = "4.0*x[1]*(0.41-x[1])/0.1681"
        U0_str = "4.*Umax*x[1]*({0}-x[1])/pow({0}, 2)".format(H)
        inlet = Expression(U0_str, degree=2, Umax=Umax)
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

    def update_bcs(self, t):
        Umax = 1.5 * np.sin(np.pi * t / 8)
        # U0_str = "4.*{1}*x[1]*({0}-x[1])/pow({0}, 2)".format(H, Umax)
        U0_str = "4.*Umax*x[1]*({0}-x[1])/pow({0}, 2)".format(H)
        t2 = datetime.now()
        inlet = Expression(U0_str, degree=2, Umax=Umax)
        t3 = datetime.now()
        bc00 = DirichletBC(self.V, inlet, self.mf, self.bc_dict["inlet"])
        self.bcs["u0"][0] = bc00
        dt = (t3 - t2).total_seconds()
        if dt > 0.1:
            print(t, t3 - t2)
        return

    def normal_stresses(self):
        p = self.q_["p"]
        n = FacetNormal(self.mesh)
        grad_u = grad(self.u_)
        tau = self.mu * (grad_u + grad_u.T) - p * Identity(2)
        F_drag = assemble(dot(tau, n)[0] * self.ds_(2))
        F_lift = assemble(dot(tau, n)[1] * self.ds_(2))
        return -self.coeff * F_drag, -self.coeff * F_lift

    def start_timestep_hook(self, t, **kvargs):
        """Called at start of new timestep"""
        self.update_bcs(t)
        # TODO: predict p_
        # X_approx, X_approx_n = self.ROM.predict(xi)
        # self.q_["p"].vector().vec().array = X_approx.ravel()
        pass

    def temporal_hook(self, t, tstep, **kvargs):
        # u = self.q_["u0"].compute_vertex_values(mesh)  # 2805
        u = self.q_["u0"].vector().vec().array  # 10942
        v = self.q_["u1"].vector().vec().array
        p = self.q_["p"].vector().vec().array
        i = tstep - 1
        pth = self.temp_dir
        self.C_D[i], self.C_L[i] = self.normal_stresses()
        self.t_u[i] = t
        if (i == 0) or ((tstep % 100) == 0):
            fig, (ax1, ax2) = self.plot()
            plt.savefig(pth + "frame_{:06d}.png".format(tstep))
            plt.close()

            pth2 = "/home/florianma@ad.ife.no/ad_disk/Florian/Repositoties/dolfinx-tutorial/chapter2/"
            turek = np.loadtxt(pth2 + "bdforces_lv4")
            turek_p = np.loadtxt(pth2 + "pointvalues_lv4")
            t_u, C_D, C_L = self.t_u, self.C_D, self.C_L
            num_velocity_dofs = len(u) + len(v)
            num_pressure_dofs = len(p)
            dofs = num_velocity_dofs + num_pressure_dofs

            fig = plt.figure(figsize=(25, 8))
            lbl = r"FEniCSx  ({0:d} dofs)".format(dofs)
            plt.plot(t_u, C_D, "-.", label=lbl, linewidth=2)
            plt.plot(
                turek[1:, 1],
                turek[1:, 3],
                marker="x",
                markevery=50,
                linestyle="",
                markersize=4,
                label="FEATFLOW (42016 dofs)",
            )
            plt.title("Drag coefficient")
            plt.grid()
            plt.legend()
            plt.savefig(pth + "drag_comparison.png")
            plt.close()

            fig = plt.figure(figsize=(25, 8))
            lbl = r"FEniCSx  ({0:d} dofs)".format(dofs)
            plt.plot(t_u, C_L, "-.", label=lbl, linewidth=2)
            plt.plot(
                turek[1:, 1],
                turek[1:, 4],
                marker="x",
                markevery=50,
                linestyle="",
                markersize=4,
                label="FEATFLOW (42016 dofs)",
            )
            plt.title("Lift coefficient")
            plt.grid()
            plt.legend()
            plt.savefig(pth + "lift_comparison.png")
            plt.close()
        np.save(pth + "u_{:06d}.npy".format(i), u)
        np.save(pth + "v_{:06d}.npy".format(i), v)
        np.save(pth + "p_{:06d}.npy".format(i), p)
        tvs = kvargs.get("tvs", None)
        if tvs:
            # TODO
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

            # fig, ax = plt.subplots()
            # plt.imshow(X_q)
            # plt.show()
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
        for nm in ["mf.xdmf", "mf.h5", "mesh.xdmf", "mesh.h5"]:
            src = self.pkg_dir + nm
            dst = pth + nm
            print(src, dst)
            copy2(src, dst)

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
