# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2021 all rights reserved
#


# framework
import ampcor


# data product foundries; these get used by the framework during component binding
@ampcor.foundry(implements=ampcor.specs.slc,
                tip="an SLC raster image")
def slc():
    # get the component
    from .SLC import SLC
    # borrow its docstring
    __doc__ = SLC.__doc__
    # and publish it
    return SLC


@ampcor.foundry(implements=ampcor.specs.offsets,
                tip="an offset map from a reference to a secondary raster")
def offsets():
    # get the component
    from .OffsetMap import OffsetMap
    # borrow its docstring
    __doc__ = OffsetMap.__doc__
    # and publish it
    return OffsetMap


# factories
def newSLC(**kwds):
    """
    Build a new SLC
    """
    # get the product
    from .SLC import SLC
    # build one and return it
    return SLC(**kwds)


def newOffsets(**kwds):
    """
    Build a new offsets map
    """
    # get the product
    from .OffsetMap import OffsetMap
    # build one and return it
    return OffsetMap(**kwds)


def newArena(**kwds):
    """
    Access an intermediate product
    """
    from .Arena import Arena
    # build one and return it
    return Arena(**kwds)


# end of file
