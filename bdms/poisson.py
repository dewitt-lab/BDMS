r"""Abstract base classes for defining general Poisson point processes on
:py:class:`bdms.TreeNode` state spaces. Several concrete child classes are included.

These classes are used to define rate-driven processes---such as birth, death, and
mutation---for simulations with :py:class:`bdms.TreeNode.evolve`.

Example
-------

>>> import bdms

Define a two-state process.

>>> poisson_process = bdms.poisson.DiscreteProcess({"a": 1.0, "b": 2.0})

Sample waiting times for each state.

>>> for state in poisson_process.rates:  # doctest: +ELLIPSIS
...     print(state, poisson_process.waiting_time_rv(state, 0.0, seed=0))
a 0.6799...
b 0.3399...
"""

from __future__ import annotations
from typing import Any, Hashable, TYPE_CHECKING
from collections.abc import Mapping, Sequence
from abc import ABC, abstractmethod
import scipy.integrate as integrate
import numpy as np
import scipy.optimize as optimize

# imports that are only used for type hints
if TYPE_CHECKING:
    import bdms

# NOTE: sphinx is currently unable to present this in condensed form when the
#       sphinx_autodoc_typehints extension is enabled
# TODO: use ArrayLike in various phenotype/time methods (current float types)
#       once it is available in a stable release
# from numpy.typing import ArrayLike
from numpy.typing import NDArray


class Process(ABC):
    r"""Abstract base class for Poisson point processes on :py:class:`bdms.TreeNode`
    attributes.

    Args:
        attr: The name of the :py:class:`bdms.TreeNode` attribute to access.
    """

    @abstractmethod
    def __init__(self, attr: str = "state") -> None:
        self.attr = attr

    def __call__(self, node: bdms.TreeNode) -> NDArray[np.float64]:
        r"""Call ``self`` to evaluate the Poisson intensity at a tree node.

        Args:
            node: The node whose state is accessed to evaluate the process.

        Returns:
            The Poisson intensity at the node.
        """
        return self.λ(getattr(node, self.attr), node.t)

    # @abstractmethod
    # def __add__(self, other: Process) -> Process:
    #     r"""The superposition of this Poisson process with another.

    #     Args:
    #         other: A Poisson process to add.
    #     """

    # @abstractmethod
    # def __rmul__(self, scaling: float):
    #     r"""Multiply a Poisson process by a constant, returning a process with a
    #     rescaled intensity measure."""

    @abstractmethod
    def λ(self, x: Hashable, t: float) -> NDArray[np.float64]:
        r"""The Poisson intensity :math:`\lambda(x, t)` for state :math:`x` at time
        :math:`t`.

        Args:
            x: State to evaluate Poisson intensity at.
            t: Time to evaluate Poisson intensity at (``0.0`` corresponds to
               the root). This only has an effect if the process is time-inhomogeneous.

        Returns:
            The Poisson intensity :math:`\lambda(x, t)`.
        """

    @abstractmethod
    def Λ(self, x: Hashable, t: float, Δt: float) -> NDArray[np.float64]:
        r"""Evaluate the Poisson intensity measure of state :math:`x` and time interval
        :math:`[t, t+Δt)`, defined as.

        .. math::     \Lambda(x, t, Δt) = \int_{t}^{t+Δt} \lambda(x, s)ds,

        This is needed for sampling waiting times and evaluating the log probability
        density function of waiting times.

        Args:
            x: State to evaluate Poisson intensity measure at.
            t: Start time.
            Δt: Time interval duration.

        Returns:
            The Poisson intensity measure :math:`\Lambda(x, t, \Delta t)`.
        """

    @abstractmethod
    def Λ_inv(self, x: Hashable, t: float, τ: float) -> NDArray[np.float64]:
        r"""Evaluate the inverse function wrt :math:`\Delta t` of :py:meth:`Process.Λ`,
        :math:`\Lambda_t^{-1}(x, t, \tau)`, such that :math:`\Lambda_t^{-1}(x, t,
        \Lambda(x, t, t+\Delta t)) = \Delta t`. This is needed for sampling waiting
        times. Note that :math:`\Lambda_t^{-1}` is well-defined iff :math:`\lambda(x, t)
        > 0`.

        Args:
            x: State.
            t: Start time of the interval.
            τ: Poisson intensity measure of the interval.

        Returns:
            The inverse Poisson intensity measure :math:`\Lambda_t^{-1}(x, t, \tau)`.
        """

    def waiting_time_rv(
        self,
        x: Hashable,
        t: float,
        rate_multiplier: float = 1.0,
        seed: int | np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        r"""Sample the waiting time :math:`\Delta t` until the first event, given the
        process on state :math:`x` starting at time :math:`t`.

        Args:
            x: State.
            t: Time at which to start waiting.
            rate_multiplier: A constant by which to multiply the Poisson intensity
            seed: A seed to initialize the random number generation. If ``None``, then
                  fresh, unpredictable entropy will be pulled from the OS. If an
                  ``int``, then it will be used to derive the initial state. If a
                  :py:class:`numpy.random.Generator`, then it will be used directly.

        Returns:
            Waiting time :math:`\Delta t`.
        """
        rng = np.random.default_rng(seed)
        return self.Λ_inv(x, t, rng.exponential(scale=1 / rate_multiplier))

    def __repr__(self) -> str:
        keyval_strs = (
            f"{key}={value}"
            for key, value in vars(self).items()
            if not key.startswith("_")
        )
        return f"{self.__class__.__name__}({', '.join(keyval_strs)})"


class HomogeneousProcess(Process):
    r"""Abstract base class for homogenous Poisson processes."""

    @abstractmethod
    def λ_homogeneous(
        self, x: Hashable | Sequence[Hashable] | NDArray[Any]
    ) -> NDArray[np.float64]:
        r"""Evaluate homogeneous Poisson intensity :math:`\lambda(x)` for state
        :math:`x`.

        Args:
            x: State to evaluate Poisson intensity at.

        Returns:
            The Poisson intensity :math:`\lambda(x)`.
        """

    # def __add__(self, other: Process) -> Process:

    # def __rmul__(self, scaling: float):

    def λ(self, x: Hashable, t: float) -> NDArray[np.float64]:
        return self.λ_homogeneous(x)

    def Λ(self, x: Hashable, t: float, Δt: float) -> NDArray[np.float64]:
        return self.λ_homogeneous(x) * Δt

    # @np.errstate(divide="ignore")
    # NOTE: the above suppresses warnings, but is slow!
    # We instead test for zero.
    def Λ_inv(self, x: Hashable, t: float, τ: float) -> NDArray[np.float64]:
        rate = self.λ_homogeneous(x)
        if rate == 0:
            return np.full_like(x, np.inf, dtype=float)

        return τ / rate


class ConstantProcess(HomogeneousProcess):
    r"""A process with a specified constant rate (independent of state).

    Args:
        value: Constant rate.
    """

    def __init__(self, value: float = 1.0):
        super().__init__()
        self.value = value

    def λ_homogeneous(
        self, x: Hashable | Sequence[Hashable] | NDArray[Any]
    ) -> NDArray[np.float64]:
        return self.value * np.ones_like(x)


class DiscreteProcess(HomogeneousProcess):
    r"""A homogeneous process at each of :math:`d` states.

    Args:
        rates: Rates for each state.
        attr: The name of the :py:class:`bdms.TreeNode` attribute to access. This
              should take a discrete set of values.
    """

    def __init__(
        self, rates: Mapping[Hashable, float] | Sequence[float], attr: str = "state"
    ):
        super().__init__(attr=attr)
        self.rates = rates

    def λ_homogeneous(
        self, x: Hashable | Sequence[Hashable] | NDArray[Any]
    ) -> NDArray[np.float64]:
        if isinstance(x, Hashable):  # type:ignore
            return self.rates[x]  # type:ignore
        return np.array([self.rates[xi] for xi in x])


class InhomogeneousProcess(Process):
    r"""Abstract base class for homogenous Poisson processes.

    Default implementations of :py:meth:`Λ` and :py:meth:`Λ_inv` use quadrature and
    root-finding, respectively.
    You may wish to override these methods in a child clasee for better performance,
    if analytical forms are available.

    Args:
        attr: The name of the :py:class:`bdms.TreeNode` attribute to access.
        quad_kwargs: Quadrature convergence arguments passed to
                     :py:func:`scipy.integrate.quad`.
        root_kwargs: Root-finding convergence arguments passed to
                     :py:func:`scipy.optimize.root_scalar`.
    """

    def __init__(
        self,
        attr: str = "state",
        quad_kwargs: dict[str, Any] = {},
        root_kwargs: dict[str, Any] = {},
    ):
        super().__init__(attr=attr)
        self.quad_kwargs = quad_kwargs
        self.root_kwargs = root_kwargs

    @abstractmethod
    def λ_inhomogeneous(self, x: Hashable, t: float) -> NDArray[np.float64]:
        r"""Evaluate inhomogeneous Poisson intensity :math:`\lambda(x, t)` given state
        :math:`x`.

        Args:
            x: Attribute value to evaluate Poisson intensity at.
            t: Time at which to evaluate Poisson intensity.

        Returns:
            The Poisson intensity :math:`\lambda(x, t)`.
        """

    def λ(self, x: Hashable, t: float) -> NDArray[np.float64]:
        return self.λ_inhomogeneous(x, t)

    def Λ(self, x: Hashable, t: float, Δt: float) -> NDArray[np.float64]:
        return integrate.quad(lambda Δt: self.λ(x, t + Δt), 0, Δt, **self.quad_kwargs)[
            0
        ]

    def Λ_inv(self, x: Hashable, t: float, τ: float) -> NDArray[np.float64]:
        # NOTE: we log transform to ensure non-negative values
        def f(logΔt: float):
            return self.Λ(x, t, np.exp(logΔt)) - τ

        def fprime(logΔt: float):
            return self.λ(x, t + np.exp(logΔt)) * np.exp(logΔt)

        sol = optimize.root_scalar(
            f=f,
            fprime=fprime,
            x0=np.log(τ / self.λ(x, t)),  # initial guess based on rate at time t
            **self.root_kwargs,
        )
        if not sol.converged:
            raise RuntimeError(f"Root-finding failed to converge:\n{sol}")
        return np.exp(sol.root)
