# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2025 all rights reserved
#


# framework
import ampcor
# my protocol
from .Functor import Functor


# declaration
class Constant(ampcor.component,
               family="ampcor.correlators.functors.constant", implements=Functor):
    """
    A functor that add a constant offset
    """


    # user configurable state
    shift = ampcor.properties.tuple(schema=ampcor.properties.int())
    shift.default = (0,0)
    shift.doc = "the shift to apply to points"


    # protocol obligations
    @ampcor.export
    def eval(self, points, **kwds):
        """
        Map the given set of {points} to their images under my transformation
        """
        # grab my {shift} and convert to an index
        shift = ampcor.libpyre.grid.Index2D(index=self.shift)
        # invoke my implementation to compute the pairings
        pairings = ampcor.libampcor.constantShift(points=points, shift=shift)
        # and return the shifted points
        return pairings


    # interface
    def show(self, indent, margin):
        """
        Display my configuration
        """
        # show who i am
        yield f"{margin}functor: {self.pyre_family()}"
        yield f"{margin}{indent}name: {self.pyre_name}"
        yield f"{margin}{indent}family: {self.pyre_family()}"
        yield f"{margin}{indent}shift: {self.shift}"
        # all done
        return


# end of file
