# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# framework
import ampcor


# declaration
class Functor(ampcor.protocol, family="ampcor.correlators.functors"):
    """
    The protocol implemented by generators of points for the secondary raster
    """


    # requirements
    @ampcor.provides
    def eval(self, points, **kwds):
        """
        Map the given set of {points} to their images under my transformation
        """


    # framework hooks
    @classmethod
    def pyre_default(cls, **kwds):
        """
        Provide a default implementation
        """
        # pull the default implementation
        from .Constant import Constant
        # and publish it
        return Constant


# end of file
