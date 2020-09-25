# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# the dispatcher
def dispatcher(**kwds):
    """
    The handler of {uri} requests
    """
    # get the dispatcher
    from .Dispatcher import Dispatcher
    # instantiate and return
    return Dispatcher(**kwds)


# end of file
