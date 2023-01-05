# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2023 all rights reserved
#


# the package
import ampcor


# declaration
class Arena(ampcor.flow.specification, family="ampcor.products.arena"):
    """
    Access to the data of an intermediate product
    """


    # public data
    origin = ampcor.properties.tuple(schema=ampcor.properties.int())
    origin.doc = "the origin of the raster in pixels"

    shape = ampcor.properties.tuple(schema=ampcor.properties.int())
    shape.doc = "the shape of the raster in pixels"

    data = ampcor.properties.path()
    data.doc = "the path to my binary data"


    # framework hooks
    @classmethod
    def pyre_default(cls, **kwds):
        """
        Provide access to the reference implementation
        """
        # invoke the foundry and publish the product
        return ampcor.products.arena()


# end of file
