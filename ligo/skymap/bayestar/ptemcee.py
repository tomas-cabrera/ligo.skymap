# Copyright (C) 2018  Leo Singer
#
# FIXME: Remove this file if https://github.com/willvousden/ptemcee/pull/6
# is merged
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
import numpy as np
import ptemcee.sampler

__all__ = ('Sampler',)


class VectorLikePriorEvaluator(ptemcee.sampler.LikePriorEvaluator):

    def __call__(self, x):
        s = x.shape
        x = x.reshape((-1, x.shape[-1]))

        lp = self.logp(x, *self.logpargs, **self.logpkwargs)
        if np.any(np.isnan(lp)):
            raise ValueError('Prior function returned NaN.')

        ll = np.empty_like(lp)
        bad = (lp == -np.inf)
        ll[bad] = 0
        ll[~bad] = self.logl(x[~bad], *self.loglargs, **self.loglkwargs)
        if np.any(np.isnan(ll)):
            raise ValueError('Log likelihood function returned NaN.')

        return ll.reshape(s[:-1]), lp.reshape(s[:-1])


class Sampler(ptemcee.sampler.Sampler):
    """Patched version of :class:`ptemcee.Sampler` that supports the
    `vectorize` option of :class:`emcee.EnsembleSampler`."""

    def __init__(self, nwalkers, dim, logl, logp,
                 ntemps=None, Tmax=None, betas=None,
                 threads=1, pool=None, a=2.0,
                 loglargs=[], logpargs=[],
                 loglkwargs={}, logpkwargs={},
                 adaptation_lag=10000, adaptation_time=100,
                 random=None, vectorize=False):
        super().__init__(nwalkers, dim, logl, logp,
                         ntemps=ntemps, Tmax=Tmax, betas=betas,
                         threads=threads, pool=pool, a=a, loglargs=loglargs,
                         logpargs=logpargs, loglkwargs=loglkwargs,
                         logpkwargs=logpkwargs, adaptation_lag=adaptation_lag,
                         adaptation_time=adaptation_time, random=random)
        self._vectorize = vectorize
        if vectorize:
            self._likeprior = VectorLikePriorEvaluator(logl, logp,
                                                       loglargs, logpargs,
                                                       loglkwargs, logpkwargs)

    def _evaluate(self, ps):
        if self._vectorize:
            return self._likeprior(ps)
        else:
            return super(self).evaluate(ps)
