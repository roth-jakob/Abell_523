import nifty8 as ift
import numpy as np


class SkyBeamer(ift.LinearOperator):
    """Maps from the total sky domain to individual sky domains and applies the
    primary beam pattern.

    Parameters
    ----------
    domain : RGSpace
        Two-dimensional RG-Space which serves as domain. The distances are
        in pseudo-radian.

    beam_directions : dict(key: beam)
        Dictionary with beam patterns (in same RGSpace as domain)
    """

    def __init__(self, domain, beam_directions):
        self._bd = dict(beam_directions)
        self._domain = ift.makeDomain(domain)

        t, b = {}, {}
        for kk, vv in self._bd.items():
            print("\r" + kk, end="")
            t[kk], b[kk] = self._domain, vv['beam']
            assert t[kk] == b[kk].domain
        print()

        self._beams = b
        self._target = ift.makeDomain(t)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES:
            res = {}
            for kk in self._target.keys():
                res[kk] = x * self._beams[kk].val
        else:
            res = np.zeros(self._domain.shape)
            for kk, xx in x.items():
                res += xx * self._beams[kk].val
        return ift.makeField(self._tgt(mode), res)
