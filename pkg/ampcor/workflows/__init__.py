# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2021 all rights reserved
#


# support
import ampcor


# the ampcor foundry
@ampcor.foundry(tip="compute an offset map from a reference to a secondary raster",
                implements=ampcor.specs.ampcor)
def ampcor():
    """
    Compute an offset map from a reference to a secondary raster
    """
    # get the flow
    from .Ampcor import Ampcor
    # and publish it
    return Ampcor


# end of file
