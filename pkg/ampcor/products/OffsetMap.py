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
    shape.default = (0,0)
    shape.doc = "the shape of the map"

    data = ampcor.properties.path()
    data.doc = "the path to my binary data"

    # public data
    @property
    def layout(self):
        """
        Get my layout
        """
        # ask the spec
        return self.spec.layout


    @property
    def bytesPerCell(self):
        """
        Get the memory footprint of my cell
        """
        # ask the spec
        return self.spec.bytesPerCell


    # protocol obligations
    @ampcor.export
    def cells(self):
        """
        Compute the number of points
        """
        # ask my spec; it knows
        return self.spec.cells


    @ampcor.export
    def bytes(self):
        """
        Compute my memory footprint
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
        Map me over the contents of {filename}
        """
        # if we are opening in read-only mode
        if mode == "r":
            # make a const raster
            raster = ampcor.libampcor.OffsetsConstRaster(shape=self.shape, uri=self.data)
        # if we are opening an existing one in read/write mode
        elif mode == "w":
            # make a modifiable raster
            raster = ampcor.libampcor.OffsetsRaster(shape=self.shape, uri=self.data, new=False)
        # if we are creating one
        elif mode == "n":
            # make a new raster; careful: this deletes existing products
            raster = ampcor.libampcor.OffsetsRaster(shape=self.shape, uri=self.data, new=True)
        # otherwise
        else:
            # grab the journal
            import journal
            # make a channel
            channel = journal.error("ampcor.products.slc")
            # and complain
            channel.line(f"unknown mode '{mode}'")
            channel.line(f"  while opening '{self.data}'")
            channel.line(f"  in ampcor.products.OffsetMap.open();")
            channel.line(f"  valid modes are: 'r', 'w', 'n'")
            channel.log()
            # just in case errors are non-fatal
            raster = None

        # attach the raster
        self.raster = raster

        # all done
        return self


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # load my product spec
        self.spec = ampcor.libampcor.Offsets(shape=self.shape)
        # i get a raster after {open}
        self.raster = None
        # all done
        return


    def __getitem__(self, idx):
        """
        Return the pair of correlated points stored at {index}
        """
        # ask the raster
        return self.raster[idx]


    def __setitem__(self, idx, points):
        """
        Establish a correlation between the reference and secondary {points} at {index}
        """
        # delegate to the raster
        self.raster[idx] = points
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
        yield f"{margin}points: {self.cells()}"
        yield f"{margin}footprint: {self.bytes()} bytes"
        # all done
        return


# end of file
