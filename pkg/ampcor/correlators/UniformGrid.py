# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2023 all rights reserved
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
        # convert the shape into a grid shape
        shape = ampcor.libpyre.grid.Shape2D(shape=shape)
        # repeat for the raster bounds
        bounds = ampcor.libpyre.grid.Shape2D(shape=bounds)
        # compute the pairings
        domain = ampcor.libampcor.uniformGrid(bounds=bounds, shape=shape)
        # all done
        return domain


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
