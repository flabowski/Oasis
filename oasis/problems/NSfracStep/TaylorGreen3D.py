__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from oasis.problems import (
    constrained_domain,
    scalar_components,
    Schmidt,
    Schmidt_T,
    body_force,
    initialize,
    scalar_hook,
    scalar_source,
    pre_solve_hook,
    theend_hook,
    get_problem_parameters,
    post_import_problem,
    info_blue,
    create_bcs,
)
from oasis.problems.NSfracStep import (
    velocity_tentative_hook,
    pressure_hook,
    start_timestep_hook,
    temporal_hook,
    default_parameters,
)

# from oasis.problems.TaylorGreen3D import mesh
from dolfin import (
    pi,
    near,
    dot,
    assemble,
    grad,
    dx,
    inner,
    Expression,
    SubDomain,
    plot,
    BoxMesh,
    Point,
)
import numpy as np


def get_problem_parameters(**kwargs):
    # Override some problem specific parameters
    NS_parameters = dict(
        scalar_components=scalar_components,
        Schmidt=Schmidt,
        Schmidt_T=Schmidt_T,
        nu=0.005,
        T=0.2,
        dt=0.01,
        Nx=33,
        Ny=33,
        Nz=33,
        folder="taylorgreen3D_results",
        max_iter=1,
        velocity_degree=1,
        save_step=10000,
        checkpoint=10000,
        plot_interval=10,
        print_dkdt_info=10000,
        use_krylov_solvers=True,
        krylov_solvers=dict(monitor_convergence=True),
    )
    # set default parameters
    for key, val in default_parameters.items():
        if key not in NS_parameters.keys():
            NS_parameters[key] = val

    NS_expressions = dict(
        constrained_domain=PeriodicDomain(),
        kin=np.zeros(1),
        initial_fields=dict(
            u0="sin(x[0])*cos(x[1])*cos(x[2])",
            u1="-cos(x[0])*sin(x[1])*cos(x[2])",
            u2="0",
            p="1./16.*(cos(2*x[0])+cos(2*x[1]))*(cos(2*x[2])+2)",
        ),
    )
    return NS_parameters, NS_expressions


def mesh(Nx, Ny, Nz, **params):
    return BoxMesh(Point(-pi, -pi, -pi), Point(pi, pi, pi), Nx, Ny, Nz)


def near(x, y, tol=1e-12):
    return bool(abs(x - y) < tol)


class PeriodicDomain(SubDomain):
    def inside(self, x, on_boundary):
        return bool(
            (near(x[0], -pi) or near(x[1], -pi) or near(x[2], -pi))
            and (not (near(x[0], pi) or near(x[1], pi) or near(x[2], pi)))
            and on_boundary
        )

    def map(self, x, y):
        if near(x[0], pi) and near(x[1], pi) and near(x[2], pi):
            y[0] = x[0] - 2.0 * pi
            y[1] = x[1] - 2.0 * pi
            y[2] = x[2] - 2.0 * pi
        elif near(x[0], pi) and near(x[1], pi):
            y[0] = x[0] - 2.0 * pi
            y[1] = x[1] - 2.0 * pi
            y[2] = x[2]
        elif near(x[1], pi) and near(x[2], pi):
            y[0] = x[0]
            y[1] = x[1] - 2.0 * pi
            y[2] = x[2] - 2.0 * pi
        elif near(x[1], pi):
            y[0] = x[0]
            y[1] = x[1] - 2.0 * pi
            y[2] = x[2]
        elif near(x[0], pi) and near(x[2], pi):
            y[0] = x[0] - 2.0 * pi
            y[1] = x[1]
            y[2] = x[2] - 2.0 * pi
        elif near(x[0], pi):
            y[0] = x[0] - 2.0 * pi
            y[1] = x[1]
            y[2] = x[2]
        else:  # near(x[2], pi):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - 2.0 * pi


def initialize(q_, q_1, q_2, VV, initial_fields, OasisFunction, **NS_namespace):
    for ui in q_:
        vv = OasisFunction(
            Expression((initial_fields[ui]), element=VV[ui].ufl_element()), VV[ui]
        )
        vv()
        q_[ui].vector()[:] = vv.vector()[:]
        if not ui == "p":
            q_1[ui].vector()[:] = q_[ui].vector()[:]
            q_2[ui].vector()[:] = q_[ui].vector()[:]


def temporal_hook(
    u_,
    p_,
    tstep,
    plot_interval,
    print_dkdt_info,
    nu,
    dt,
    t,
    oasis_memory,
    kin,
    **NS_namespace
):
    oasis_memory("tmp", True)
    if tstep % print_dkdt_info == 0 or tstep % print_dkdt_info == 1:
        kinetic = assemble(0.5 * dot(u_, u_) * dx) / (2 * pi) ** 3
        if tstep % print_dkdt_info == 0:
            kin[0] = kinetic
            dissipation = assemble(nu * inner(grad(u_), grad(u_)) * dx) / (2 * pi) ** 3
            info_blue("Kinetic energy = {} at time = {}".format(kinetic, t))
            info_blue("Energy dissipation rate = {}".format(dissipation))
        else:
            info_blue("dk/dt = {} at time = {}".format((kinetic - kin[0]) / dt, t))

    if tstep % plot_interval == 0:
        plot(p_, title="pressure")
        plot(u_[0], title="velocity-x")
        plot(u_[1], title="velocity-y")
