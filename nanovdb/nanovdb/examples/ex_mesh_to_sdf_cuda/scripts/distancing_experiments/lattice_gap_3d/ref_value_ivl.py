"""Exact rational interval arithmetic (Fractions).  Referee's own."""
from fractions import Fraction as F

class I:
    __slots__ = ('lo', 'hi')
    def __init__(self, lo, hi=None):
        if hi is None: hi = lo
        self.lo = F(lo); self.hi = F(hi)
        assert self.lo <= self.hi
    def __add__(a, b):
        b = mk(b); return I(a.lo + b.lo, a.hi + b.hi)
    __radd__ = __add__
    def __neg__(a): return I(-a.hi, -a.lo)
    def __sub__(a, b):
        b = mk(b); return I(a.lo - b.hi, a.hi - b.lo)
    def __rsub__(a, b): return mk(b) - a
    def __mul__(a, b):
        b = mk(b)
        p = (a.lo*b.lo, a.lo*b.hi, a.hi*b.lo, a.hi*b.hi)
        return I(min(p), max(p))
    __rmul__ = __mul__
    def __truediv__(a, b):
        b = mk(b)
        if b.lo <= 0 <= b.hi: raise ZeroDivisionError("interval contains 0")
        p = (a.lo/b.lo, a.lo/b.hi, a.hi/b.lo, a.hi/b.hi)
        return I(min(p), max(p))
    def __repr__(a): return "[%.20g, %.20g]" % (float(a.lo), float(a.hi))
    def certainly_pos(a): return a.lo > 0
    def certainly_neg(a): return a.hi < 0
    def certainly_lt(a, c): return a.hi < c
    def certainly_ge(a, c): return a.lo >= c

def mk(x):
    return x if isinstance(x, I) else I(x)

def isqrt_floor(n):
    if n < 0: raise ValueError
    if n == 0: return 0
    x = 1 << ((n.bit_length() + 1) // 2)
    while True:
        y = (x + n // x) // 2
        if y >= x: return x
        x = y

_SCALE = 10**80
def SQRT(n):
    """Interval enclosing sqrt(n) for integer n >= 0."""
    N = n * _SCALE * _SCALE
    r = isqrt_floor(N)
    lo = F(r, _SCALE); hi = F(r+1, _SCALE)
    assert lo*lo <= n <= hi*hi
    return I(lo, hi)
