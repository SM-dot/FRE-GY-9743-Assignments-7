import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)



class Interpolator1DMonotoneConvex(Interpolator1D):
    """Shape-preserving C1 quadratic spline for IFR nodes (Schumaker-style).

    Inside each node interval [t_i, t_{i+1}] (width h, secant m = (f_{i+1} - f_i) / h):

    1. Node slopes d_i: a weighted average of the two neighbouring secants
       (the slope of the parabola through three nodes). If the neighbouring
       secants have opposite signs (a local max/min at the node), d_i = 0.
       End slopes: d_0 = 2 m_0 - d_1, d_n = 2 m_{n-1} - d_{n-1}, set to 0
       if that would point the wrong way.
    2. With a = m - d_i and b = d_{i+1} - m, if a > 0 and b > 0 we insert one
       extra knot at xi = t_i + h * b / (a + b). The slope then moves linearly
       d_i -> m on [t_i, xi] and m -> d_{i+1} on [xi, t_{i+1}]. So the curve is two
       quadratic pieces, its slope only goes up (convex), the slope never
       changes sign (monotone), it hits both node values and is C1.
    3. If a < 0 or b < 0 (data is not locally convex, e.g. an inflection or
       a concave stretch), a convex C1 piece is impossible. We then use a
       straight line on that interval: still continuous, still through the
       nodes, monotone and (trivially) convex; only smoothness at the nodes
       is given up.

    Outside [t_0, t_n] we extrapolate FLAT. A single node gives a constant.
    The curve is piecewise quadratic, so `integrate` is exact.
    """

    _TOL = 1e-14

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT
        t = np.asarray(self.axis1_, dtype=float)
        f = np.asarray(self.values_, dtype=float)
        assert self.length_ >= 1
        assert self.length_ == 1 or np.all(np.diff(t) > 0), "nodes must be strictly increasing"
        self.t_ = t
        self.f_ = f

        # Each segment k: f(x) = v[k] + s[k] * dx + q[k] * dx^2, dx = x - x[k], on [x[k], x[k+1]].
        seg_x, seg_v, seg_s, seg_q = [], [], [], []
        if self.length_ > 1:
            h = np.diff(t)
            m = np.diff(f) / h
            d = self._node_slopes(h, m)
            for i in range(self.length_ - 1):
                a = m[i] - d[i]
                b = d[i + 1] - m[i]
                if a > self._TOL and b > self._TOL:
                    xi = h[i] * b / (a + b)
                    seg_x += [t[i], t[i] + xi]
                    seg_v += [f[i], f[i] + xi * (d[i] + m[i]) / 2.0]
                    seg_s += [d[i], m[i]]
                    seg_q += [(m[i] - d[i]) / (2.0 * xi), (d[i + 1] - m[i]) / (2.0 * (h[i] - xi))]
                else:
                    # Linear fallback (exact C1 when a == b == 0).
                    seg_x.append(t[i])
                    seg_v.append(f[i])
                    seg_s.append(m[i])
                    seg_q.append(0.0)
            seg_x.append(t[-1])  # right edge of the last segment

        self.seg_x_ = np.asarray(seg_x, dtype=float)
        self.seg_v_ = np.asarray(seg_v, dtype=float)
        self.seg_s_ = np.asarray(seg_s, dtype=float)
        self.seg_q_ = np.asarray(seg_q, dtype=float)
        # Cumulative integral from t_0 to the start of each segment (and to t_n at the end).
        widths = np.diff(self.seg_x_)
        seg_int = self.seg_v_ * widths + self.seg_s_ * widths**2 / 2.0 + self.seg_q_ * widths**3 / 3.0
        self.cum_int_ = np.concatenate(([0.0], np.cumsum(seg_int)))

    @staticmethod
    def _node_slopes(h: np.ndarray, m: np.ndarray) -> np.ndarray:
        n = len(m) + 1
        d = np.zeros(n)
        for i in range(1, n - 1):
            if m[i - 1] * m[i] > 0:
                d[i] = (h[i] * m[i - 1] + h[i - 1] * m[i]) / (h[i - 1] + h[i])
        d[0] = 2.0 * m[0] - d[1] if n > 2 else m[0]
        d[-1] = 2.0 * m[-1] - d[-2] if n > 2 else m[-1]
        if d[0] * m[0] < 0:
            d[0] = 0.0
        if d[-1] * m[-1] < 0:
            d[-1] = 0.0
        return d

    def interpolate(self, x: float) -> float:
        if self.length_ == 1 or x <= self.t_[0]:
            return float(self.f_[0])
        if x >= self.t_[-1]:
            return float(self.f_[-1])
        k = int(np.searchsorted(self.seg_x_, x, side='right')) - 1
        dx = x - self.seg_x_[k]
        return float(self.seg_v_[k] + self.seg_s_[k] * dx + self.seg_q_[k] * dx * dx)

    def _antiderivative(self, x: float) -> float:
        """Integral of the curve from t_0 to x (negative for x < t_0)."""
        if self.length_ == 1 or x <= self.t_[0]:
            return float(self.f_[0] * (x - self.t_[0]))
        if x >= self.t_[-1]:
            return float(self.cum_int_[-1] + self.f_[-1] * (x - self.t_[-1]))
        k = int(np.searchsorted(self.seg_x_, x, side='right')) - 1
        dx = x - self.seg_x_[k]
        return float(self.cum_int_[k] + self.seg_v_[k] * dx
                     + self.seg_s_[k] * dx**2 / 2.0 + self.seg_q_[k] * dx**3 / 3.0)

    def integrate(self, start_x: float, end_x: float) -> float:
        return self._antiderivative(end_x) - self._antiderivative(start_x)

    # Optional risk interface
    # The test cells for risk are not prepared but you can do it by your own.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        raise NotImplementedError("Optional risk: ordinate gradient is not implemented")

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        raise NotImplementedError("Optional risk: integrated ordinate gradient is not implemented")


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')
