import numpy as np
import sympy as sp
from scipy import sparse
from scipy.integrate import quad
from numpy.polynomial import Legendre as Leg, Chebyshev as Cheb

from typing import Literal, Optional, Callable, Self
from abc import ABC, abstractmethod

type Domain = tuple[float, float]
type Boundary = tuple[float, float]
type Function = sp.Function | Callable[[np.ndarray], np.ndarray]

x = sp.Symbol("x")


def map_reference_domain(x: float, d: Domain, r: Domain) -> float:
    """Map the physical domain to the reference domain

    Args:
        x (float): Point in the physical domain
        d (Domain): Physical domain
        r (Domain): Reference domain

    Returns:
        np.ndarray: Reference domain
    """
    return r[0] + (r[1] - r[0]) * (x - d[0]) / (d[1] - d[0])


def map_true_domain(x: float, d: Domain, r: Domain) -> float:
    """Map the reference domain to the physical domain

    Args:
        x (float): Point in the reference domain
        d (Domain): Physical domain
        r (Domain): Reference domain

    Returns:
        np.ndarray: Physical domain
    """
    return d[0] + (d[1] - d[0]) * (x - r[0]) / (r[1] - r[0])


def map_expression_true_domain(
    u: sp.Expr, x: sp.Symbol, d: Domain, r: Domain
) -> sp.Expr:
    """Map the reference domain to the physical domain in an expression

    Args:
        u (sp.Expr): Expression
        x (sp.Symbol): Variable
        d (Domain): Physical domain
        r (Domain): Reference domain

    Returns:
        sp.Expr: Mapped expression
    """
    if d != r:
        u = sp.sympify(u)
        xm = map_true_domain(x, d, r)
        u = u.replace(x, xm)
    return u


class FunctionSpace(ABC):
    """Base class for function spaces"""

    def __init__(self, N: int, domain: Domain = (-1, 1)) -> None:
        """Initialize the function space

        Args:
            N (int): Number of basis functions
            domain (Domain, optional): Physical domain. Defaults to (-1, 1).
        """
        self.N = N
        self._domain = domain

    @property
    def domain(self) -> Domain:
        """Return the physical domain of the function space"""
        return self._domain

    @property
    @abstractmethod
    def reference_domain(self) -> Domain:
        """Return the reference domain of the function space"""
        pass

    @property
    def domain_factor(self) -> float:
        """Return the factor to scale the domain"""
        d = self.domain
        r = self.reference_domain
        return (d[1] - d[0]) / (r[1] - r[0])

    def mesh(self, N: Optional[int] = None) -> np.ndarray:
        """Return the mesh of the function space.

        Args:
            N (int, optional): Number of elements. Defaults to None.

        Returns:
            np.ndarray: Mesh of the function space
        """
        d = self.domain
        n = N if N is not None else self.N
        return np.linspace(d[0], d[1], n + 1)

    def weight(self, x: sp.Symbol = x) -> Literal[1]:
        """Return the weight function of the function space

        Args:
            x (sp.Symbol, optional): Variable. Defaults to x.

        Returns:
            Literal[1]: Weight function
        """
        return 1

    @abstractmethod
    def basis_function(self, j: int, sympy: bool = False) -> Function:
        """Return the j-th basis function of the function space

        Args:
            j (int): Basis function index
            sympy (bool, optional): Whether to use sympy expressions. Defaults to False.

        Returns:
            Callable: Basis function
        """
        pass

    @abstractmethod
    def derivative_basis_function(self, j: int, k: int = 1) -> Function:
        """Return the k-th derivative of the j-th basis function of the function space

        Args:
            j (int): Basis function index
            k (int, optional): Derivative order. Defaults to 1.

        Returns:
            Callable: Derivative of the basis function
        """
        pass

    def evaluate_basis_function(self, Xj: np.ndarray, j: int) -> np.ndarray:
        """Evaluate the j-th basis function at the points Xj

        Args:
            Xj (np.ndarray): Points in the reference domain
            j (int): Basis function index

        Returns:
            np.ndarray: Basis function values
        """
        return self.basis_function(j)(Xj)

    def evaluate_derivative_basis_function(
        self, Xj: np.ndarray, j: int, k: int = 1
    ) -> np.ndarray:
        """Evaluate the k-th derivative of the j-th basis function at the points Xj

        Args:
            Xj (np.ndarray): Points in the reference domain
            j (int): Basis function index
            k (int, optional): Derivative order. Defaults to 1.

        Returns:
            np.ndarray: Derivative of the basis function values
        """
        return self.derivative_basis_function(j, k=k)(Xj)

    def eval(self, uh: np.ndarray, xj: np.ndarray) -> np.ndarray:
        """Evaluate the function at the points xj

        Args:
            uh (np.ndarray): Function coefficients
            xj (np.ndarray): Points in the physical domain

        Returns:
            np.ndarray: Function values
        """
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        P = self.eval_basis_function_all(Xj)
        return P @ uh

    def eval_basis_function_all(self, Xj: np.ndarray) -> np.ndarray:
        """Evaluate all basis functions at the points Xj

        Args:
            Xj (np.ndarray): Points in the reference domain

        Returns:
            np.ndarray: Basis function values
        """
        P = np.zeros((len(Xj), self.N + 1))
        for j in range(self.N + 1):
            P[:, j] = self.evaluate_basis_function(Xj, j)
        return P

    def eval_derivative_basis_function_all(
        self, Xj: np.ndarray, k: int = 1
    ) -> np.ndarray:
        """Evaluate all derivatives of the basis functions at the points Xj

        Args:
            Xj (np.ndarray): Points in the reference domain
            k (int, optional): Derivative order. Defaults to 1.

        Returns:
            np.ndarray: Derivative of the basis function values
        """
        raise NotImplementedError

    def inner_product(self, u: sp.Expr) -> np.ndarray:
        """Compute the inner product of the expression u with the basis functions

        Args:
            u (sp.Expr): Expression

        Returns:
            np.ndarray: Inner product
        """
        us = map_expression_true_domain(u, x, self.domain, self.reference_domain)
        us: Callable[[np.ndarray], np.ndarray] = sp.lambdify(x, us)
        uj = np.zeros(self.N + 1)
        h = self.domain_factor
        r = self.reference_domain
        for i in range(self.N + 1):
            psi = self.basis_function(i)

            def uv(Xj: np.ndarray) -> np.ndarray:
                return us(Xj) * psi(Xj)

            uj[i] = float(h) * quad(uv, float(r[0]), float(r[1]))[0]
        return uj

    def mass_matrix(self) -> np.ndarray:
        """Compute the mass matrix of the function space

        Returns:
            np.ndarray: Mass matrix
        """
        return assemble_generic_matrix(TrialFunction(self), TestFunction(self))


class Legendre(FunctionSpace):
    """Base class for Legendre function spaces"""

    def __init__(self, N: int, domain: Domain = (-1, 1)) -> None:
        """Initialize the function space

        Args:
            N (int): Number of basis functions
            domain (Domain, optional): Physical domain. Defaults to (-1, 1).
        """
        FunctionSpace.__init__(self, N, domain=domain)

    def basis_function(self, j: int, sympy: bool = False) -> sp.legendre | Leg:
        """Return the j-th basis function of the function space

        Args:
            j (int): Basis function index
            sympy (bool, optional): Whether to use sympy expressions. Defaults to False.

        Returns:
            Callable: Basis function
        """
        if sympy:
            return sp.legendre(j, x)
        return Leg.basis(j)

    @property
    def reference_domain(self) -> Domain:
        return (-1, 1)

    def derivative_basis_function(self, j: int, k: int = 1) -> sp.legendre | Leg:
        """Return the k-th derivative of the j-th basis function of the function space

        Args:
            j (int): Basis function index
            k (int, optional): Derivative order. Defaults to 1.

        Returns:
            Callable: Derivative of the basis function
        """
        return self.basis_function(j).deriv(k)

    def L2_norm_sq(self, N: int) -> float:
        """Return the square of the L2 norm of the basis functions

        Args:
            N (int): Number of basis functions

        Returns:
            float: L2 norm squared
        """
        return 2 / (2 * N + 1)

    def mass_matrix(self) -> np.ndarray:
        """Compute the mass matrix of the function space

        Returns:
            np.ndarray: Mass matrix
        """
        arr = np.vectorize(self.L2_norm_sq)(np.arange(self.N + 1))
        return sparse.diags_array(arr, shape=(self.N + 1, self.N + 1))

    def eval(self, uh: np.ndarray, xj: np.ndarray) -> np.ndarray:
        """Evaluate the function at the points xj

        Args:
            uh (np.ndarray): Function coefficients
            xj (np.ndarray): Points in the physical domain

        Returns:
            np.ndarray: Function values
        """
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        return np.polynomial.legendre.legval(Xj, uh)


class Chebyshev(FunctionSpace):
    """Base class for Chebyshev function spaces"""

    def __init__(self, N: int, domain: Domain = (-1, 1)) -> None:
        """Initialize the function space

        Args:
            N (int): Number of basis functions
            domain (Domain, optional): Physical domain. Defaults to (-1, 1).
        """
        FunctionSpace.__init__(self, N, domain=domain)

    def basis_function(self, j: int, sympy: bool = False) -> sp.Function | Cheb:
        """Return the j-th basis function of the function space

        Args:
            j (int): Basis function index
            sympy (bool, optional): Whether to use sympy expressions. Defaults to False.

        Returns:
            Callable: Basis function
        """
        if sympy:
            return sp.cos(j * sp.acos(x))
        return Cheb.basis(j)

    @property
    def reference_domain(self) -> Domain:
        return (-1, 1)

    def derivative_basis_function(self, j: int, k: int = 1) -> sp.Function | Cheb:
        """Return the k-th derivative of the j-th basis function of the function space

        Args:
            j (int): Basis function index
            k (int, optional): Derivative order. Defaults to 1.

        Returns:
            Callable: Derivative of the basis function
        """
        return self.basis_function(j).deriv(k)

    def weight(self, x: sp.Symbol = x) -> sp.Expr:
        return 1 / sp.sqrt(1 - x**2)

    def L2_norm_sq(self, N: int) -> float:
        """Return the square of the L2 norm of the basis functions

        # Isn't this the weighted L2 norm?

        Args:
            N (int): Number of basis functions

        Returns:
            float: L2 norm squared
        """
        c = 2 if N == 0 else 1
        return c * np.pi / 2

    def mass_matrix(self) -> np.ndarray:
        """Compute the mass matrix of the Chebyshev function space

        Returns:
            np.ndarray: Mass matrix
        """
        arr = np.vectorize(self.L2_norm_sq)(np.arange(self.N + 1))
        return sparse.diags_array(arr, shape=(self.N + 1, self.N + 1))

    def eval(self, uh: np.ndarray, xj: np.ndarray) -> np.ndarray:
        """Evaluate the function at the points xj

        Args:
            uh (np.ndarray): Function coefficients
            xj (np.ndarray): Points in the physical domain

        Returns:
            np.ndarray: Function values
        """
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        return np.polynomial.chebyshev.chebval(Xj, uh)

    def inner_product(self, u: sp.Expr) -> np.ndarray:
        """Compute the inner product of the expression u with the basis functions

        Args:
            u (sp.Expr): Expression

        Returns:
            np.ndarray: Inner product
        """
        us = map_expression_true_domain(u, x, self.domain, self.reference_domain)
        # change of variables to x=cos(theta)
        us = sp.simplify(us.subs(x, sp.cos(x)), inverse=True)
        us = sp.lambdify(x, us)
        uj = np.zeros(self.N + 1)
        h = float(self.domain_factor)
        k = sp.Symbol("k")
        basis = sp.lambdify(
            (k, x),
            sp.simplify(self.basis_function(k, True).subs(x, sp.cos(x), inverse=True)),
        )
        for i in range(self.N + 1):

            def uv(Xj, j):
                return us(Xj) * basis(j, Xj)

            uj[i] = float(h) * quad(uv, 0, np.pi, args=(i,))[0]
        return uj


class Trigonometric(FunctionSpace):
    """Base class for trigonometric function spaces"""

    @property
    def reference_domain(self) -> Domain:
        return (0, 1)

    def mass_matrix(self) -> sparse.dia_array:
        """Compute the mass matrix of the function space

        Returns:
            np.ndarray: Mass matrix
        """
        arr = np.vectorize(self.L2_norm_sq)(np.arange(self.N + 1))
        return sparse.diags_array(arr, shape=(self.N + 1, self.N + 1))

    def eval(self, uh: np.ndarray, xj: np.ndarray) -> np.ndarray:
        """Evaluate the function at the points xj

        Args:
            uh (np.ndarray): Function coefficients
            xj (np.ndarray): Points in the physical domain

        Returns:
            np.ndarray: Function values
        """
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        P = self.eval_basis_function_all(Xj)
        return P @ uh + self.B.Xl(Xj)

    @property
    def B(self) -> "BoundaryCondition":
        return self._B

    @B.setter
    def B(self, B: "BoundaryCondition") -> None:
        self._B = B

    @abstractmethod
    def L2_norm_sq(self, N: int) -> float:
        pass


class Sines(Trigonometric):
    """Function space of sines"""

    def __init__(self, N: int, domain: Domain = (0, 1), bc: Boundary = (0, 0)) -> None:
        """Initialize the function space

        Args:
            N (int): Number of basis functions
            domain (Domain, optional): Physical domain. Defaults to (0, 1).
            bc (Boundary, optional): Boundary conditions. Defaults to (0, 0).
        """
        Trigonometric.__init__(self, N, domain=domain)
        self.B = Dirichlet(bc, domain, self.reference_domain)

    def basis_function(self, j: int, sympy: bool = False) -> Function:
        """Return the j-th basis function of the function space

        Args:
            j (int): Basis function index
            sympy (bool, optional): Whether to use sympy expressions. Defaults to False.

        Returns:
            Callable: Basis function
        """
        if sympy:
            return sp.sin((j + 1) * sp.pi * x)
        return lambda Xj: np.sin((j + 1) * np.pi * Xj)

    def derivative_basis_function(
        self, j: int, k: int = 1
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Return the k-th derivative of the j-th basis function of the function space

        Args:
            j (int): Basis function index
            k (int, optional): Derivative order. Defaults to 1.

        Returns:
            Callable: Derivative of the basis function
        """
        scale = ((j + 1) * np.pi) ** k * {0: 1, 1: -1}[(k // 2) % 2]
        if k % 2 == 0:
            return lambda Xj: scale * np.sin((j + 1) * np.pi * Xj)
        else:
            return lambda Xj: scale * np.cos((j + 1) * np.pi * Xj)

    def L2_norm_sq(self, N: int) -> float:
        return 0.5


class Cosines(Trigonometric):

    def __init__(self, N: int, domain: Domain = (0, 1), bc: Boundary = (0, 0)) -> None:
        Trigonometric.__init__(self, N, domain=domain)
        self.B = Neumann(bc, domain, self.reference_domain)

    def basis_function(self, j: int, sympy: bool = False) -> Function:
        if sympy:
            return sp.cos(j * sp.pi * x)
        return lambda Xj: np.cos(j * np.pi * Xj)

    def derivative_basis_function(
        self, j: int, k: int = 1
    ) -> Callable[[np.ndarray], np.ndarray]:
        scale = (j * np.pi) ** k
        func = [np.cos, np.sin][k % 2]
        sign = [1, -1, -1, 1][k % 4]

        return lambda Xj: sign * scale * func(j * np.pi * Xj)

    def L2_norm_sq(self, N: int) -> float:
        return 1.0 if N == 0 else 0.5


# Create classes to hold the boundary function


class BoundaryCondition(ABC):
    """Base class for boundary conditions"""

    def __init__(self, bc: Boundary, domain: Domain, reference_domain: Domain) -> None:
        """Initialize the boundary condition

        Args:
            bc (Boundary): Boundary conditions
            domain (Domain): Physical domain
            reference_domain (Domain): Reference domain
        """
        self.bc = bc
        self.domain = domain
        self.reference_domain = reference_domain

        self.x = self._map_to_physical_domain()
        self.xX = map_expression_true_domain(
            self.x, x, self.domain, self.reference_domain
        )
        self.Xl = sp.lambdify(x, self.xX)

    @abstractmethod
    def _map_to_physical_domain(self) -> sp.Expr:
        """Map the reference domain to the physical domain

        Returns:
            sp.Expr: Mapped expression
        """
        pass


class Dirichlet(BoundaryCondition):

    def _map_to_physical_domain(self) -> sp.Expr:
        d0, d1 = self.domain
        bc0, bc1 = self.bc
        h = d1 - d0
        return bc0 * (d1 - x) / h + bc1 * (x - d0) / h


class Neumann(BoundaryCondition):
    def _map_to_physical_domain(self) -> sp.Expr:
        d0, d1 = self.domain
        bc0, bc1 = self.bc
        h = d1 - d0
        return bc0 / h * (d1 * x - x**2 / 2) + bc1 / h * (x**2 / 2 - d0 * x)


class Composite(FunctionSpace):
    r"""Base class for function spaces created as linear combinations of orthogonal basis functions

    The composite basis functions are defined using the orthogonal basis functions
    (Chebyshev or Legendre) and a stencil matrix S. The stencil matrix S is used
    such that basis function i is

    .. math::

        \psi_i = \sum_{j=0}^N S_{ij} Q_j

    where :math:`Q_i` can be either the i'th Chebyshev or Legendre polynomial

    For example, both Chebyshev and Legendre have Dirichlet basis functions

    .. math::

        \psi_i = Q_i-Q_{i+2}

    Here the stencil matrix will be

    .. math::

        s_{ij} = \delta_{ij} - \delta_{i+2, j}, \quad (i, j) \in (0, 1, \ldots, N) \times (0, 1, \ldots, N+2)

    Note that the stencil matrix is of shape :math:`(N+1) \times (N+3)`.
    """

    def eval(self, uh: np.ndarray, xj: np.ndarray) -> np.ndarray:
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        P = self.eval_basis_function_all(Xj)
        return P @ uh + self.B.Xl(Xj)

    def mass_matrix(self):
        arr = np.vectorize(self.L2_norm_sq)(np.arange(self.N + 3))
        M = sparse.diags_array(arr, shape=(self.N + 3, self.N + 3))
        return self.S @ M @ self.S.T

    @property
    def S(self) -> sparse.csr_array:
        return self._S

    @S.setter
    def S(self, S: sparse.csr_array) -> None:
        self._S = S


class DirichletLegendre(Composite, Legendre):
    def __init__(self, N, domain=(-1, 1), bc=(0, 0)):
        Legendre.__init__(self, N, domain=domain)
        self.B = Dirichlet(bc, domain, self.reference_domain)
        self.S = sparse.diags_array([1, -1], offsets=[0, 2], shape=(N + 1, N + 3))

    def basis_function(self, j: int, sympy: bool = False) -> Function:
        if sympy:
            return sp.legendre(j, x) - sp.legendre(j + 2, x)

        return Leg.basis(j) - Leg.basis(j + 2)


class NeumannLegendre(Composite, Legendre):
    def __init__(
        self,
        N: int,
        domain: Domain = (-1, 1),
        bc: Boundary = (0, 0),
        constraint: float = 0,
    ) -> None:
        Legendre.__init__(self, N, domain=domain)
        self.constraint = constraint
        self.B = Neumann(bc, domain, self.reference_domain)
        arr = np.arange(N + 1)
        self.S = sparse.diags_array(
            [
                (arr + 2) * (arr + 3),
                -arr * (arr + 1),
            ],
            offsets=[0, 2],
            shape=(N + 1, N + 3),
        )

    def basis_function(self, j: int, sympy: bool = False) -> Function:
        def factors(A, B):
            return A * (j + 2) * (j + 3) - B * (j + 1) * j

        if sympy:
            return factors(sp.legendre(j, x), sp.legendre(j + 2, x))
        return factors(Leg.basis(j), Leg.basis(j + 2))


class DirichletChebyshev(Composite, Chebyshev):

    def __init__(self, N: int, domain: Domain = (-1, 1), bc: Boundary = (0, 0)) -> None:
        Chebyshev.__init__(self, N, domain=domain)
        self.B = Dirichlet(bc, domain, self.reference_domain)
        self.S = sparse.diags_array([1, -1], offsets=[0, 2], shape=(N + 1, N + 3))

    def basis_function(self, j: int, sympy: bool = False) -> Function:
        if sympy:
            return sp.cos(j * sp.acos(x)) - sp.cos((j + 2) * sp.acos(x))
        return Cheb.basis(j) - Cheb.basis(j + 2)


class NeumannChebyshev(Composite, Chebyshev):
    def __init__(
        self,
        N: int,
        domain: Domain = (-1, 1),
        bc: Boundary = (0, 0),
        constraint: float = 0,
    ) -> None:
        Chebyshev.__init__(self, N, domain=domain)
        self.constraint = constraint
        self.B = Neumann(bc, domain, self.reference_domain)
        self.S = sparse.diags_array(
            [
                (np.arange(N + 1) + 2) ** 2,
                -np.arange(N + 1) ** 2,
            ],
            offsets=[0, 2],
            shape=(N + 1, N + 3),
        )

    def basis_function(self, j: int, sympy: bool = False) -> Function:
        def factors(A, B):
            return A * (j + 2) ** 2 - B * j**2

        if sympy:
            return factors(sp.cos(j * sp.acos(x)), sp.cos((j + 2) * sp.acos(x)))
        return factors(Cheb.basis(j), Cheb.basis(j + 2))


class BasisFunction:

    def __init__(self, V: FunctionSpace, diff: int = 0, argument=0) -> None:
        self._V = V
        self._num_derivatives = diff
        self._argument = argument

    @property
    def argument(self):
        return self._argument

    @property
    def function_space(self) -> FunctionSpace:
        return self._V

    @property
    def num_derivatives(self) -> int:
        return self._num_derivatives

    def diff(self, k) -> Self:
        return self.__class__(self.function_space, diff=self.num_derivatives + k)


class TestFunction(BasisFunction):

    def __init__(self, V: FunctionSpace, diff: int = 0) -> None:
        BasisFunction.__init__(self, V, diff=diff, argument=0)


class TrialFunction(BasisFunction):

    def __init__(self, V: FunctionSpace, diff: int = 0) -> None:
        BasisFunction.__init__(self, V, diff=diff, argument=1)


def assemble_generic_matrix(u, v) -> np.ndarray:
    assert isinstance(u, TrialFunction)
    assert isinstance(v, TestFunction)
    V = v.function_space
    assert u.function_space == V
    r = V.reference_domain
    D = np.zeros((V.N + 1, V.N + 1))
    cheb = V.weight() == 1 / sp.sqrt(1 - x**2)
    symmetric = True if u.num_derivatives == v.num_derivatives else False
    w = {"weight": "alg" if cheb else None, "wvar": (-0.5, -0.5) if cheb else None}

    def uv(Xj, i, j):
        return V.evaluate_derivative_basis_function(
            Xj, i, k=v.num_derivatives
        ) * V.evaluate_derivative_basis_function(Xj, j, k=u.num_derivatives)

    for i in range(V.N + 1):
        for j in range(i if symmetric else 0, V.N + 1):
            D[i, j] = quad(uv, float(r[0]), float(r[1]), args=(i, j), **w)[0]
            if symmetric:
                D[j, i] = D[i, j]
    return D


def inner(u, v: TestFunction) -> np.ndarray:
    V = v.function_space
    h = V.domain_factor
    if isinstance(u, TrialFunction):
        num_derivatives = u.num_derivatives + v.num_derivatives
        if num_derivatives == 0:
            return float(h) * V.mass_matrix()
        else:
            return float(h) ** (1 - num_derivatives) * assemble_generic_matrix(u, v)
    return V.inner_product(u)


def project(ue, V):
    u = TrialFunction(V)
    v = TestFunction(V)
    b = inner(ue, v)
    A = inner(u, v)
    uh = sparse.linalg.spsolve(A, b)
    return uh


def L2_error(uh, ue, V, kind="norm"):
    d = V.domain
    uej = sp.lambdify(x, ue)

    def uv(xj):
        return (uej(xj) - V.eval(uh, xj)) ** 2

    return np.sqrt(quad(uv, float(d[0]), float(d[1]))[0])


def test_project():
    ue = sp.besselj(0, x)
    domain = (0, 10)
    for space in (Chebyshev, Legendre):
        V = space(16, domain=domain)
        u = project(ue, V)
        err = L2_error(u, ue, V)
        print(f"test_project: L2 error = {err:2.4e}, N = {V.N}, {V.__class__.__name__}")
        assert err < 1e-6


def test_helmholtz():
    ue = sp.besselj(0, x)
    f = ue.diff(x, 2) + ue
    domain = (0, 10)
    for space in (
        NeumannChebyshev,
        NeumannLegendre,
        DirichletChebyshev,
        DirichletLegendre,
        Sines,
        Cosines,
    ):
        if space in (NeumannChebyshev, NeumannLegendre, Cosines):
            bc = ue.diff(x, 1).subs(x, domain[0]), ue.diff(x, 1).subs(x, domain[1])
        else:
            bc = ue.subs(x, domain[0]), ue.subs(x, domain[1])
        N = 60 if space in (Sines, Cosines) else 12
        V = space(N, domain=domain, bc=bc)
        u = TrialFunction(V)
        v = TestFunction(V)
        A = inner(u.diff(2), v) + inner(u, v)
        b = inner(f - (V.B.x.diff(x, 2) + V.B.x), v)
        u_tilde = np.linalg.solve(A, b)
        err = L2_error(u_tilde, ue, V)
        print(f"test_helmholtz: L2 error = {err:2.4e}, N = {N}, {V.__class__.__name__}")
        assert err < 1e-3


def test_convection_diffusion():
    eps = 0.05
    ue = (sp.exp(-x / eps) - 1) / (sp.exp(-1 / eps) - 1)
    f = 0
    domain = (0, 1)
    for space in (DirichletLegendre, DirichletChebyshev, Sines):
        N = 50 if space is Sines else 16
        V = space(N, domain=domain, bc=(0, 1))
        u = TrialFunction(V)
        v = TestFunction(V)
        A = inner(u.diff(2), v) + (1 / eps) * inner(u.diff(1), v)
        b = inner(f - ((1 / eps) * V.B.x.diff(x, 1)), v)
        u_tilde = np.linalg.solve(A, b)
        err = L2_error(u_tilde, ue, V)
        print(
            f"test_convection_diffusion: L2 error = {err:2.4e}, N = {N}, {V.__class__.__name__}"
        )
        assert err < 1e-3


if __name__ == "__main__":
    test_project()
    test_convection_diffusion()
    test_helmholtz()
