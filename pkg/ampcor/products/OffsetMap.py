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


# declaration
class OffsetMap(ampcor.flow.product,
                family="ampcor.products.offsets.offsets", implements=ampcor.specs.offsets):
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
    def cells(self):
        """
        Compute the number of points
        """


    @ampcor.export
    def bytes(self):
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
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # all done
        return


    def __getitem__(self, index):
        """
        Return the pair of correlated points stored at {index}
        """


    def __setitem__(self, index, points):
        """
        Establish a correlation between the reference and secondary {points} at {index}
        """


    # implementation details
    def show(self, indent, margin):
        """
        Generate a report of my configuration
        """
        # my info
        yield f"{margin}name: {self.pyre_name}"
        yield f"{margin}family: {self.pyre_family()}"
        yield f"{margin}data: {self.data}"
        yield f"{margin}shape: {self.shape}"
        yield f"{margin}points: {self.cells()}"
        yield f"{margin}footprint: {self.bytes()} bytes"
        # all done
        return


# end of file
