# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
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
    A domain that generates domain points on a uniform grid
    """


    # protocol requirements
    @ampcor.export
    def points(self, shape, bounds, **kwds):
        """
        Generate a cloud of points within {bounds} where reference tiles will be placed
        """
        # split {bounds} into evenly spaced tiles
        tile = tuple(b//s for b,s in zip(bounds, shape))
        # compute the unallocated border around the raster
        margin = tuple(b%s for b,s in zip(bounds, shape))
        # build the sequences of coordinates for tile centers along each axis
        ticks = tuple(
            # by generating the locations
            tuple(m//2 + n*t + t//2 for n in range(g))
            # given the layout of each axis
            for g, m, t in zip(shape, margin, tile)
        )
        # their cartesian product generates the centers of all the tiles in the grid
        centers = tuple(itertools.product(*ticks))
        # all done
        return centers


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
