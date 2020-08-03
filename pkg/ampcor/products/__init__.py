# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# framework
import ampcor


# data product foundries; these get used by the framework during component binding
@ampcor.foundry(implements=raster, tip="an SLC raster image")
def slc():
    # get the component
    from .SLC import SLC
    # borrow its docstring
    __doc__ = SLC.__doc__
    # and publish it
    return SLC


@ampcor.foundry(implements=raster, tip="an offset map from a reference to a secondary raster")
def offsets():
    # get the component
    from .OffsetMap import OffsetMap
    # borrow its docstring
    __doc__ = OffsetMap.__doc__
    # and publish it
    return OffsetMap


# end of file
