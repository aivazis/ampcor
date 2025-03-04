# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2025 all rights reserved
#


# the framework
import ampcor


# declaration
class SLC(ampcor.flow.product,
          family="ampcor.products.slc.slc", implements=ampcor.specs.slc):
    """
    Access to the data of a file based SLC
    """


    # user configurable state
    shape = ampcor.properties.tuple(schema=ampcor.properties.int())
    shape.default = (0,0)
    shape.doc = "the shape of the raster in pixels"

    data = ampcor.properties.path()
    data.doc = "the path to my binary data"


    # public data
    # the memory footprint of an individual pixel; the {SLC} extension knows...
    bytesPerPixel = ampcor.libampcor.SLC.bytesPerPixel

    @property
    def layout(self):
        """
        Get my layout
        """
        # ask the spec
        return self.spec.layout


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
        # go through each rank
        for o,s, l in zip(origin, shape, self.shape):
            # and check
            if o < 0 : return
            if o+s >= l: return
        # ask my spec
        return self.spec.slice(origin, shape)


    @ampcor.export
    def open(self, mode="r"):
        """
        Map me over the contents of my {data} file
        """
        # unpack the shape
        shape = self.shape

        # attempt to
        try:
            # resolve the filename using the {vfs}
            uri = self.pyre_fileserver[self.data].uri
        # if that fails
        except Exception:
            # use the raw name
            uri = self.data

        # if we are opening in read-only mode
        if mode == "r":
            # make a const raster
            raster = ampcor.libampcor.SLCConstRaster(shape=shape, uri=uri)
        # if we are opening an existing one in read/write mode
        elif mode == "w":
            # make a modifiable raster
            raster = ampcor.libampcor.SLCRaster(shape=shape, uri=uri, new=False)
        # if we are creating one
        elif mode == "n":
            # make a new raster; careful: this deletes existing products
            raster = ampcor.libampcor.SLCRaster(shape=shape, uri=uri, new=True)
        # otherwise
        else:
            # grab the journal
            import journal
            # make a channel
            channel = journal.error("ampcor.products.slc")
            # and complain
            channel.line(f"unknown mode '{mode}'")
            channel.line(f"  while opening '{uri}'")
            channel.line(f"  in ampcor.products.SLC.open();")
            channel.line(f"  valid modes are: 'r', 'w', 'n'")
            channel.log()
            # just in case errors are non-fatal
            raster = None

        # attach the raster
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


    def __getitem__(self, idx):
        # ask the raster
        return self.raster[idx]


    def __setitem__(self, idx, pixel):
        # delegate to the raster
        self.raster[idx] = pixel
        # all done
        return


    # implementation details
    def show(self, indent, margin):
        """
        Generate a report of my configuration
        """
        # my info
        yield f"{margin}slc:"
        yield f"{margin}{indent}name: {self.pyre_name}"
        yield f"{margin}{indent}family: {self.pyre_family()}"
        yield f"{margin}{indent}data: {self.data}"
        yield f"{margin}{indent}shape: {self.shape}"
        yield f"{margin}{indent}pixels: {self.cells():,}"
        yield f"{margin}{indent}footprint: {self.bytes()/1024**3:,.3f} Gb"
        # all done
        return


# end of file
