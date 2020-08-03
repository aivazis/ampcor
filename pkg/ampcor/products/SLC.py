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
class SLC(ampcor.flow.product,
          family="ampcor.products.slc.slc", implements=ampcor.specs.slc):
    """
    Access to the data of a file based SLC
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
        Compute the number of pixels
        """


    @ampcor.export
    def bytes(self):
        """
        Compute my memory footprint, in bytes
        """


    @ampcor.export
    def slice(self, origin, shape):
        """
        Grant access to a slice of data of the given {shape} starting at {origin}
        """


    @ampcor.export
    def open(self, mode="r"):
        """
        Map me over the contents of my {data} file
        """


    # metamethods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # all done
        return


# end of file
