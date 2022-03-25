__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-06"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from dolfin import inner, dx, grad, dot, nabla_grad, assemble, norm, normalize
from dolfin import Matrix, Vector, Function, VectorSpaceBasis, Timer
from dolfin import PETScPreconditioner, PETScKrylovSolver, LUSolver
from dolfin import as_backend_type, as_vector
import oasis.common.utilities as ut
import matplotlib.pyplot as plt


def attach_pressure_nullspace(Ap, p, Q):
    """Create null space basis object and attach to Krylov solver."""
    null_vec = Vector(p)
    Q.dofmap().set(null_vec, 1.0)
    null_vec *= 1.0 / null_vec.norm("l2")
    Aa = as_backend_type(Ap)
    null_space = VectorSpaceBasis([null_vec])
    Aa.set_nullspace(null_space)
    Aa.null_space = null_space


class FirstInner:
    def __init__(self, domain):
        u, v = domain.u, domain.v
        dmn = self.domain = domain
        # - - - - - - - - - - - - -SETUP- - - - - - - - - - - - - - - - - -
        # Mass matrix
        M = ut.assemble_matrix(inner(u, v) * dx)
        # Stiffness matrix (without viscosity coefficient)
        K = ut.assemble_matrix(inner(grad(u), grad(v)) * dx)
        # Allocate stiffness matrix for LES that changes with time
        KT = (
            None
            if dmn.les_model == "NoModel" and dmn.nn_model == "NoModel"
            else (Matrix(M), inner(grad(u), grad(v)))
        )
        # Pressure Laplacian.
        # Allocate coefficient matrix (needs reassembling)
        A = Matrix(M)
        # Setup for solving convection
        dim = len(dmn.u_components)
        u_ab = as_vector([Function(dmn.V) for i in range(dim)])
        a_conv = inner(v, dot(u_ab, nabla_grad(u))) * dx
        a_scalar = a_conv
        LT = (
            None
            if dmn.les_model == "NoModel"
            else ut.LESsource(dmn.nut_, u_ab, dmn.V, name="LTd")
        )

        NT = (
            None
            if dmn.nn_model == "NoModel"
            else ut.NNsource(dmn.nunn_, u_ab, dmn.V, name="NTd")
        )
        # for first iter:
        self.A = A
        self.a_conv = a_conv
        self.M = M
        self.a_scalar = a_scalar
        self.K = K
        self.LT = LT
        self.KT = KT
        self.NT = NT
        self.u_ab = u_ab
        return

    def assemble(self):
        """Called on first inner iteration of velocity/pressure system.

        Assemble convection matrix, compute rhs of tentative velocity and
        reset coefficient matrix for solve.
        """
        dmn = self.domain
        A = self.A
        a_conv = self.a_conv
        M = self.M
        K = self.K
        LT = self.LT
        KT = self.KT
        NT = self.NT
        u_ab = self.u_ab

        t0 = Timer("Assemble first inner iter")

        # Update u_ab used as convecting velocity
        for i, ui in enumerate(dmn.u_components):
            u_ab[i].vector().zero()
            u_ab[i].vector().axpy(1.5, dmn.q_1[ui].vector())
            u_ab[i].vector().axpy(-0.5, dmn.q_2[ui].vector())
        # does not need to be assembled. matrix multipl. is enough
        # a_conv from init: inner(v, dot(u_ab, nabla_grad(u))) * dx
        A = assemble(a_conv, tensor=A)
        A *= -0.5  # Negative convection on the rhs
        A.axpy(1.0 / dmn.dt, M, True)  # Add mass
        # Set up scalar matrix for rhs using the same convection as velocity
        # ... we want to do that in the ScalarSolver.assemble() instead

        # Add diffusion and compute rhs for all velocity components
        A.axpy(-0.5 * dmn.nu, K, True)  # TODO
        if dmn.les_model != "NoModel":
            assemble(dmn.nut_ * KT[1] * dx, tensor=KT[0])
            A.axpy(-0.5, KT[0], True)

        if dmn.nn_model != "NoModel":
            assemble(dmn.nunn_ * KT[1] * dx, tensor=KT[0])
            A.axpy(-0.5, KT[0], True)

        for i, ui in enumerate(dmn.u_components):
            # Start with body force b0
            # TODO: dmn.b_tmp[ui].assign(dmn.b0[ui])
            dmn.b_tmp[ui].zero()
            dmn.b_tmp[ui].axpy(1.0, dmn.b0[ui])
            # Add transient, convection and diffusion
            dmn.b_tmp[ui].axpy(1.0, A * dmn.q_1[ui].vector())
            if dmn.les_model != "NoModel":
                LT.assemble_rhs(i)
                dmn.b_tmp[ui].axpy(1.0, LT.vector())
            if dmn.nn_model != "NoModel":
                NT.assemble_rhs(i)
                dmn.b_tmp[ui].axpy(1.0, NT.vector())

        # Reset matrix for lhs
        A *= -1.0
        A.axpy(2.0 / dmn.dt, M, True)
        [bc.apply(A) for bc in dmn.bcs["u0"]]  # TODO: is this correct?
        t0.stop()
        return


class TentativeVelocityStep:
    def __init__(self, domain):
        dmn = self.domain = domain
        # - - - - - - - - - - - - -SETUP- - - - - - - - - - - - - - - - - -
        # Allocate a dictionary of Functions for holding and computing
        # pressure gradients
        gradp = {}
        p_ = dmn.q_["p"]
        method = dmn.velocity_update_solver
        for i, ui in enumerate(dmn.u_components):
            name = "dpd" + ("x", "y", "z")[i]
            bcs = ut.homogenize(dmn.bcs[ui])
            gradp[ui] = ut.GradFunction(p_, dmn.V, i, bcs, name, method)

        # - - - - - - - - - -get_solvers - - - - - - - - - - - - - - - - - -
        if domain.use_krylov_solvers:
            vks = dmn.velocity_krylov_solver
            u_prec = PETScPreconditioner(vks["preconditioner_type"])
            u_sol = PETScKrylovSolver(vks["solver_type"], u_prec)
            u_sol.parameters.update(dmn.krylov_solvers)
        else:
            u_sol = LUSolver()
        self.u_sol = u_sol
        self.gradp = gradp
        return

    def assemble(self, ui):
        dmn = self.domain
        dmn.b[ui].zero()
        dmn.b[ui].axpy(1.0, dmn.b_tmp[ui])
        self.gradp[ui].assemble_rhs(dmn.q_["p"])
        dmn.b[ui].axpy(-1.0, self.gradp[ui].rhs)
        return

    def solve(self, ui, udiff):
        """Linear algebra solve of tentative velocity component."""
        dmn = self.domain
        [bc.apply(dmn.b[ui]) for bc in dmn.bcs[ui]]
        # q_2 only used on inner_iter 1, so use here as work vector
        dmn.q_2[ui].assign(dmn.q_[ui])
        t1 = Timer("Tentative Linear Algebra Solve")
        self.u_sol.solve(self.A, dmn.q_[ui].vector(), dmn.b[ui])
        t1.stop()
        udiff[0] += norm(dmn.q_2[ui].vector() - dmn.q_[ui].vector())
        return

    def velocity_update(self, ui):
        """Update the velocity after regular pressure velocity iterations."""
        dmn = self.domain
        # for ui in u_components:
        grad_dp = self.gradp[ui](dmn.dp_)  # grad(p_old - p_new)
        # print(grad_dp is self.gradp[ui].vector())
        # u = -dt*u + grad(dp_x); v = -dt*u + grad(dp_y)
        dmn.q_[ui].vector().axpy(-dmn.dt, grad_dp)
        [bc.apply(dmn.q_[ui].vector()) for bc in dmn.bcs[ui]]
        return


class PressureStep:
    def __init__(self, domain):
        q, p = domain.q, domain.p
        dmn = self.domain = domain
        # - - - - - - - - - - - - -SETUP- - - - - - - - - - - - - - - - - -
        # Allocate Function for holding and computing the
        # velocity divergence on Q
        method = dmn.velocity_update_solver
        divu = ut.DivFunction(dmn.u_, dmn.Q, name="divu", method=method)
        # Pressure Laplacian.
        Ap = ut.assemble_matrix(inner(grad(q), grad(p)) * dx, dmn.bcs["p"])
        if dmn.bcs["p"] == []:
            attach_pressure_nullspace(Ap, dmn.q_["p"].vector(), dmn.Q)
        if dmn.use_krylov_solvers:
            # pressure solver ##
            pks = dmn.pressure_krylov_solver
            p_prec = PETScPreconditioner(pks["preconditioner_type"])
            p_sol = PETScKrylovSolver(pks["solver_type"], p_prec)
            p_sol.parameters.update(dmn.krylov_solvers)
            p_sol.set_reuse_preconditioner(True)
        else:
            # pressure solver ##
            p_sol = LUSolver()
        self.divu = divu
        self.Ap = Ap
        self.p_sol = p_sol
        return

    def assemble(self):
        """Assemble rhs of pressure equation.
        rhs = -1/dt*inner(div(u), v) *dx + inner(grad(p*), (grad(q)) *dx"""
        dmn = self.domain
        self.divu.assemble_rhs()  # Computes div(u_)*q*dx
        dmn.b["p"][:] = self.divu.rhs
        dmn.b["p"] *= -1.0 / dmn.dt
        dmn.b["p"].axpy(1.0, self.Ap * dmn.q_["p"].vector())
        # print("divu", (self.divu.rhs.get_local() ** 2).sum() ** 0.5)
        return

    def solve(self, p_approx=False):
        """Solve pressure equation."""
        dmn = self.domain
        dpv = dmn.dp_.vector()
        p_ = dmn.q_["p"].vector()  # =p*

        [bc.apply(dmn.b["p"]) for bc in dmn.bcs["p"]]
        dpv.zero()
        dpv.axpy(1.0, p_)  # dp_ = 1*0 + p_
        # KrylovSolvers use nullspace for normalization of pressure
        if hasattr(self.Ap, "null_space"):
            self.p_sol.null_space.orthogonalize(dmn.b["p"])

        t1 = Timer("Pressure Linear Algebra Solve")
        if hasattr(p_approx, "__len__"):
            p_.array = p_approx.ravel()
        else:
            self.p_sol.solve(self.Ap, p_, dmn.b["p"])
        t1.stop()
        # LUSolver use normalize directly for normalization of pressure
        if dmn.bcs["p"] == []:
            normalize(p_)
        dpv.axpy(-1.0, p_)  # dp_ = -1*p* + p_new
        dpv *= -1.0  # dp_ = p* - p_new
        # dpv = dp_ is only used for the velocity correction
        return


class ScalarSolver:
    def __init__(self, domain):
        # TODO: M from first inner
        M = self.M  # from FirstInnerIter
        dmn = self.domain = domain
        # ... get_solvers:
        if dmn.use_krylov_solvers:
            # scalar solver ##
            if len(dmn.scalar_components) > 0:
                c_prec = PETScPreconditioner(
                    dmn.scalar_krylov_solver["preconditioner_type"]
                )
                c_sol = PETScKrylovSolver(
                    dmn.scalar_krylov_solver["solver_type"], c_prec
                )
                c_sol.parameters.update(dmn.krylov_solvers)
            else:
                c_sol = None
        else:
            if len(dmn.scalar_components) > 0:
                c_sol = LUSolver()
            else:
                c_sol = None

        # ... setup:
        # Allocate coefficient matrix and work vectors for scalars.
        # Matrix differs from velocity in boundary conditions only
        if len(dmn.scalar_components) > 0:
            self.Ta = Matrix(M)
            if len(dmn.scalar_components) > 1:
                # For more than one scalar we use the same linear algebra
                # solver for all.
                # For this to work we need some additional tensors.
                # The extra matrix is required since different scalars may have
                # different boundary conditions
                Tb = Matrix(M)
                sc0 = dmn.scalar_components[0]
                bb = Vector(dmn.q_[sc0].vector())
                bx = Vector(dmn.q_[sc0].vector())
                self.Tb = Tb
                self.bb = bb
                self.bx = bx

    def assemble(self):
        """Assemble scalar equation."""
        dmn = self.domain
        M = self.M  # mass matrix from FirstInnerIter
        K = self.K  # stiffness matrix from FirstInnerIter
        KT = self.KT  # time dep. stiffness mat for LES from FirstInnerIter

        # # # # - - - assemble_first_inner_iter - - - # # #
        # Set up scalar matrix for rhs using the same convection as velocity
        if len(dmn.scalar_components) > 0:
            Ta = self.Ta
            if self.a_scalar is self.a_conv:
                # TODO:
                Ta.zero()
                Ta.axpy(1.0, self.A, True)

        # # # # - - - scalar_assemble - - - # # #
        # Just in case you want to use a different scalar convection
        if self.a_scalar is not self.a_conv:
            assemble(self.a_scalar, tensor=Ta)
            Ta *= -0.5  # Negative convection on the rhs
            Ta.axpy(1.0 / dmn.dt, M, True)  # Add mass

        # Compute rhs for all scalars
        for ci in dmn.scalar_components:
            # Add diffusion
            Ta.axpy(-0.5 * dmn.nu / dmn.Schmidt[ci], K, True)
            if dmn.les_model != "NoModel":
                Ta.axpy(-0.5 / dmn.Schmidt_T[ci], KT[0], True)
            if dmn.nn_model != "NoModel":
                Ta.axpy(-0.5 / dmn.Schmidt_T[ci], KT[0], True)

            # Compute rhs
            dmn.b[ci].zero()
            dmn.b[ci].axpy(1.0, Ta * dmn.q_1[ci].vector())
            dmn.b[ci].axpy(1.0, dmn.b0[ci])

            # Subtract diffusion
            Ta.axpy(0.5 * dmn.nu / dmn.Schmidt[ci], K, True)
            if dmn.les_model != "NoModel":
                Ta.axpy(0.5 / dmn.Schmidt_T[ci], KT[0], True)
            if dmn.nn_model != "NoModel":
                Ta.axpy(0.5 / dmn.Schmidt_T[ci], KT[0], True)

        # Reset matrix for lhs - Note scalar matrix does not contain diffusion
        Ta *= -1.0
        Ta.axpy(2.0 / dmn.dt, M, True)
        return

    def solve(self, ci):
        """Solve scalar equation."""
        dmn = self.domain
        K = self.K  # stiffness matrix from FirstInnerIter
        Ta = self.Ta
        Ta.axpy(0.5 * dmn.nu / dmn.Schmidt[ci], K, True)  # Add diffusion
        if len(dmn.scalar_components) > 1:
            # Reuse solver for all scalars.
            # This requires the same matrix and vectors to be used by c_sol.
            Tb = self.Tb
            bb = self.bb
            bx = self.bx
            # TODO: use assign()
            Tb.zero()
            Tb.axpy(1.0, Ta, True)
            bb.zero()
            bb.axpy(1.0, dmn.b[ci])
            bx.zero()
            bx.axpy(1.0, dmn.q_[ci].vector())
            [bc.apply(Tb, bb) for bc in dmn.bcs[ci]]
            self.c_sol.solve(Tb, bx, bb)
            dmn.q_[ci].vector().zero()
            dmn.q_[ci].vector().axpy(1.0, bx)

        else:
            [bc.apply(Ta, dmn.b[ci]) for bc in dmn.bcs[ci]]
            self.c_sol.solve(Ta, dmn.q_[ci].vector(), dmn.b[ci])
        Ta.axpy(-0.5 * dmn.nu / dmn.Schmidt[ci], K, True)  # Subtract diffusion
        # x_[ci][x_[ci] < 0] = 0.               # Bounded solution
        # x_[ci].set_local(maximum(0., x_[ci].array()))
        # x_[ci].apply("insert")
        return
