# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
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
    def codomain(self, domain, **kwds):
        """
        Given a sequence of points in {domain}, generate their images
        """
        # grab my {shift}
        shift = self.shift
        # go through the points
        for point in domain:
            # apply the shift and yield the point
            yield tuple(p+s for p,s in zip(point, shift))
        # all done
        return


    # interface
    def show(self, indent, margin):
        """
        Display my configuration
        """
        # show who i am
        yield f"{margin}functor: {self.pyre_family()}"
        yield f"{margin}{indent}shift: {self.shift}"
        # all done
        return


# end of file
