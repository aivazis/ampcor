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
class OffsetMap(ampcor.component, family="ampcor.dom.rasters.offsets", implements=Raster):
    """
    Access to the data of an offset map
    """


    # public data
    shape = ampcor.properties.tuple(schema=ampcor.properties.int())
    shape.doc = "the shape of the raster in pixels"

    data = ampcor.properties.path()
    data.doc = "the path to my binary data"


    # protocol obligations
    @ampcor.export
    def capacity(self):
        """
        Compute my memory footprint
        """


    @ampcor.export
    def footprint(self):
        """
        Compute my memory footprint
        """


    @ampcor.export
    def slice(self, origin, shape):
        """
        Grant access to a slice of data of the given {shape} starting at {origin}
        """


    @ampcor.export
    def open(self, mode="r"):
        """
        Map me over the contents of {filename}
        """


    # meta-methods
    def __init__(self, shape, layout=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # storage
        self.domain = []
        self.codomain = []
        # access as a Cartesian map
        self.tile = ampcor.grid.tile(shape=shape, layout=layout)
        # all done
        return


    def __getitem__(self, index):
        """
        Return the pair of correlated points stored at {index}
        """
        # ask my tile for the offset
        offset = self.tile.offset(index)
        # pull the corresponding points
        ref = self.domain[offset]
        sec = self.codomain[offset]
        # and return them
        return (ref, sec)


    def __setitem__(self, index, points):
        """
        Establish a correlation between the reference and secondary {points} at {index}
        """
        # ask my tile for the offset
        offset = self.tile.offset(index)
        # unpack the points
        ref, sec = points
        # store them
        self.domain[offset] = ref
        self.codomain[offset] = sec
        # all done
        return


    def __len__(self):
        """
        Compute my length
        """
        # delegate to the corresponding property
        return self.size


# end of file
