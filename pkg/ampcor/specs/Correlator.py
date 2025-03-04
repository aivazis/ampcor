# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2025 all rights reserved
#


# framework
import ampcor


# declaration
class Correlator(ampcor.flow.producer, family="ampcor.correlators"):
    """
    The protocol for all AMPCOR correlator implementations
    """


    # requirements
    @ampcor.provides
    def estimate(self, **kwds):
        """
        Estimate the offset field between a pair of raster images
        """


    # hooks
    @classmethod
    def pyre_default(cls, **kwds):
        """
        Provide a default implementation
        """
        # and publish it
        return ampcor.correlators.mga()


# end of file
