# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# the framework
import ampcor


# declaration
class SLC(ampcor.flow.product,
          family="ampcor.products.slc.slc", implements=ampcor.specs.slc):
    """
    Access to the data of a file based SLC
    """


    # public data
    shape = ampcor.properties.tuple(schema=ampcor.properties.int())
    shape.default = (0,0)
    shape.doc = "the shape of the raster in pixels"

    data = ampcor.properties.path()
    data.doc = "the path to my binary data"


    # protocol obligations
    @ampcor.export
    def cells(self):
        """
        Compute the number of pixels
        """
        # ask my spec; it knows
        return self.spec.cells


    @ampcor.export
    def bytes(self):
        """
        Compute my memory footprint, in bytes
        """
        # ask my spec; it knows
        return self.spec.bytes


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
        # if we are opening in read-only mode
        if mode == "r":
            # make a const raster
            raster = ampcor.libampcor.SLCConstRaster(shape=self.shape, uri=self.data)
        # if we are opening an existing one in read/write mode
        elif mode == "w":
            # make a modifiable raster
            raster = ampcor.libampcor.SLCRaster(shape=self.shape, uri=self.data, new=False)
        # if we are creating one
        elif mode == "n":
            # make a new raster; careful: this deletes existing products
            raster = ampcor.libampcor.SLCRaster(shape=self.shape, uri=self.data, new=True)
        # otherwise
        else:
            # grab the journal
            import journal
            # make a channel
            channel = journal.error("ampcor.products.slc")
            # and complain
            channel.line(f"unknown mode '{mode}'")
            channel.line(f"  while opening '{self.data}'")
            channel.line(f"  in ampcor.products.SLC.open();")
            channel.line(f"  valid modes are: 'r', 'w', 'n'")
            channel.log()
            # just in case errors are non-fatal
            raster = None

        # make the raster and attach it
        self.raster = raster
        # all done
        return self


    # metamethods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # load my product spec
        self.spec = ampcor.libampcor.SLC(shape=self.shape)
        # i get a raster when i'm attached to a file
        self.raster = None
        # all done
        return


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
        yield f"{margin}pixels: {self.cells()}"
        yield f"{margin}footprint: {self.bytes()} bytes"
        # all done
        return


# end of file
