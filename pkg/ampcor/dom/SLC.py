# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# the framework
import ampcor
# the extension
from ampcor.ext import ampcor as libampcor
# my protocol
from .Raster import Raster


# declaration
class SLC(ampcor.component, family="ampcor.dom.rasters.slc", implements=Raster):
    """
    Access to the data of a file based SLC
    """


    # public data
    shape = ampcor.properties.tuple(schema=ampcor.properties.int())
    shape.doc = "the shape of the raster in pixels"

    data = ampcor.properties.path()
    data.doc = "the path to my binary data"


    # constants
    # the memory footprint of individual pixels
    pixelFootprint = libampcor.ConstSLC.pixelFootprint
    # factories
    from .Slice import Slice as newSlice


    # protocol obligations
    @ampcor.export
    def capacity(self):
        """
        Compute the number of pixels
        """
        # my tile knows
        return self.tile.size


    @ampcor.export
    def footprint(self):
        """
        Compute my memory footprint, in bytes
        """
        # compute and return
        return self.tile.size * self.pixelFootprint


    @ampcor.export
    def slice(self, origin, shape):
        """
        Grant access to a slice of data of the given {shape} starting at {origin}
        """
        # go through the indices that describe the {origin} of the slice
        for idx in origin:
            # if any of them are negative
            if idx < 0:
                # mark the slice as invalid
                return None
        # similarly, make sure that we don't overflow on the other end
        for beg, shp, lim in zip(origin, shape, self.shape):
            # if any of them do
            if beg + shp > lim:
                # mark this slice as invalid
                return None
        # if all goes well, make a slice and return it
        return self.newSlice(raster=self, begin=origin, shape=shape)


    @ampcor.export
    def open(self, mode="r"):
        """
        Map me over the contents of my {data} file
        """
        # build the raster
        self.raster = libampcor.ConstSLC(uri=self.data, shape=self.shape)
        # all done
        return


    # metamethods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # make a tile out of my shape
        self.tile = ampcor.grid.tile(shape=self.shape)
        # all done
        return


    # implementation details
    # private data
    raster = None  # set by {open}


# end of file
