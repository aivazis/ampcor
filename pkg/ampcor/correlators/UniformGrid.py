# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2021 all rights reserved
#


# externals
import itertools
# framework
import ampcor
# my protocol
from .Domain import Domain as domain


# declaration
class UniformGrid(ampcor.component,
                  family="ampcor.correlators.domains.uniform", implements=domain):
    """
    A strategy that generates a uniform grid
    """


    # protocol requirements
    @ampcor.export
    def points(self, shape, bounds, **kwds):
        """
        Cover {bounds} with a grid of uniformly spaced points of the given {shape}
        """
        # split {bounds} into evenly spaced tiles
        tile = (b//(s+1) for b,s in zip(bounds, shape))
        # compute the unallocated border around the raster
        margin = (b%(s+1) for b,s in zip(bounds, shape))
        # build the sequences of coordinates for tile centers along each axis
        ticks = (
            # by generating the locations
            tuple(m//2 + (n+1)*t for n in range(g))
            # given the layout of each axis
            for g, m, t in zip(shape, margin, tile)
        )
        # their cartesian product generates the centers of all the tiles in the grid
        yield from itertools.product(*ticks)
        # all done
        return


    # interface
    def show(self, indent, margin):
        """
        Display my configuration
        """
        # show who i am
        yield f"{margin}domain:"
        yield f"{margin}{indent}name: {self.pyre_name}"
        yield f"{margin}{indent}family: {self.pyre_family()}"
        # all done
        return


# end of file
