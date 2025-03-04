# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2025 all rights reserved
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


    # configurable state
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
            raster = ampcor.libampcor.OffsetsConstRaster(shape=shape, uri=uri)
        # if we are opening an existing one in read/write mode
        elif mode == "w":
            # make a modifiable raster
            raster = ampcor.libampcor.OffsetsRaster(shape=shape, uri=uri, new=False)
        # if we are creating one
        elif mode == "n":
            # make a new raster; careful: this deletes existing products
            raster = ampcor.libampcor.OffsetsRaster(shape=shape, uri=uri, new=True)
        # otherwise
        else:
            # grab the journal
            import journal
            # make a channel
            channel = journal.error("ampcor.products.slc")
            # and complain
            channel.line(f"unknown mode '{mode}'")
            channel.line(f"  while opening '{uri}'")
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


    # framework hooks
    def pyre_traitModified(self, trait, new, old):
        """
        Handle post construction configuration changes
        """
        # when my shape changes
        if trait.name == "shape":
            # recompute my spec
            self.spec = ampcor.libampcor.Offsets(shape=self.shape)
        # all done
        return self


    # implementation details
    def show(self, indent, margin):
        """
        Generate a report of my configuration
        """
        # my info
        yield f"{margin}offset map:"
        yield f"{margin}{indent}name: {self.pyre_name}"
        yield f"{margin}{indent}family: {self.pyre_family()}"
        yield f"{margin}{indent}data: {self.data}"
        yield f"{margin}{indent}shape: {self.shape}"
        yield f"{margin}{indent}points: {self.cells():,}"
        yield f"{margin}{indent}footprint: {self.bytes()/1024**2:,.3f} Mb"
        # all done
        return


# end of file
