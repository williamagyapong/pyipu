"""
PyIPU: Python Implementation of Iterative Proportional Updating

A general case of iterative proportional fitting that can satisfy two disparate sets of marginals
that do not agree on a single total. A common example is balancing population data using household-
and person-level marginal controls for survey expansion or synthetic population creation.

Based on the algorithm proposed by Ye et al. (2009) in the paper:
"Methodology to match distributions of both household and person attributes in generation of synthetic populations"
"""

from .core import ipu, ipu_matrix
from .synthesis import synthesize
from .version import __version__

__all__ = ['ipu', 'ipu_matrix', 'synthesize', '__version__']
