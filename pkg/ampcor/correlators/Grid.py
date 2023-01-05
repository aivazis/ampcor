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


# declaration
class Grid(ampcor.component,
           family="ampcor.correlators.covers.grid", implements=ampcor.correlators.cover):
    """
    A generator of a Cartesian grid of initial guesses for the offset map
    """


    # user configurable state
    domain = ampcor.correlators.domain()
    domain.doc = "the domain of the map"

    functor = ampcor.correlators.functor()
    functor.doc = "the function that maps points from the reference raster to the secondary raster"


    # requirements
    @ampcor.export
    def map(self, bounds, shape, **kwds):
        """
        Build an offset map between {reference} and {secondary}
        """
        # generate the set of points in the domain
        p = self.domain.points(bounds=bounds, shape=shape)
        # map them
        pairings = self.functor.eval(points=p)
        # pack and ship
        return pairings


    # interface
    def show(self, indent, margin):
        """
        Display my configuration
        """
        # show who i am
        yield f"{margin}initial cover generator:"
        yield f"{margin}{indent}name: {self.pyre_name}"
        yield f"{margin}{indent}family: {self.pyre_family()}"
        # show my domain
        yield from self.domain.show(indent=indent, margin=margin+indent)
        # and my codomain generator
        yield from self.functor.show(indent=indent, margin=margin+indent)
        # all done
        return


# end of file
