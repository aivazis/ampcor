# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# framework
import ampcor


# declaration
class Cover(ampcor.protocol, family="ampcor.correlators.covers"):
    """
    The protocol for initial guesses for the offset map
    """


    # requirements
    @ampcor.provides
    def map(self, **kwds):
        """
        Build an offset map between {reference} and {secondary}
        """


    # hooks
    @classmethod
    def pyre_default(cls, **kwds):
        """
        Provide a default implementation
        """
        # pull the default implementation
        from .Grid import Grid
        # and publish it
        return Grid


# end of file
