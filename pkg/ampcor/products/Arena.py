# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2021 all rights reserved
#


# the framework
import ampcor


# declaration
class Arena(ampcor.flow.product,
            family="ampcor.products.arena.arena", implements=ampcor.specs.arena):
    """
    Access to the data of an intermediate product
    """


    # user configurable state
    origin = ampcor.properties.tuple(schema=ampcor.properties.int())
    origin.default = (0,0,0)
    origin.doc = "the origin of the raster in pixels"

    shape = ampcor.properties.tuple(schema=ampcor.properties.int())
    shape.default = (0,0,0)
    shape.doc = "the shape of the raster in pixels"

    data = ampcor.properties.path()
    data.doc = "the path to my binary data"


    # types
    # my product spec
    Spec = ampcor.libampcor.Arena
    # product rasters
    ConstRaster = ampcor.libampcor.ArenaConstRaster


    # public data
    @property
    def layout(self):
        """
        Get my layout
        """
        # ask the spec
        return self.spec.layout


    @property
    def bytesPerPixel(self):
        """
        Get the memory footprint of my pixel
        """
        # ask the spec
        return self.spec.bytesPerPixel


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
        for o,s, l,h in zip(origin, shape, self.origin, self.shape):
            # and check
            if o < l : return
            if o+s > l+h: return
        # ask my spec
        return self.spec.slice(origin, shape)


    @ampcor.export
    def open(self, mode="r"):
        """
        Map me over the contents of my {data} file
        """
        # if we are opening in read-only mode
        if mode == "r":
            # make a const raster
            raster = self.ConstRaster(uri=self.data, spec=self.spec)
        # if we are opening an existing one in read/write mode
        elif mode == "w":
            # make a modifiable raster
            raster = self.Raster(uri=self.data, new=False, spec=self.spec)
        # if we are creating one
        elif mode == "n":
            # make a new raster; careful: this deletes existing products
            raster = self.Raster(uri=self.data, new=True, spec=self.spec)
        # otherwise
        else:
            # grab the journal
            import journal
            # make a channel
            channel = journal.error("ampcor.products.arena")
            # and complain
            channel.line(f"unknown mode '{mode}'")
            channel.line(f"  while opening '{self.data}'")
            channel.line(f"  in ampcor.products.Arena.open();")
            channel.line(f"  valid modes are: 'r', 'w', 'n'")
            channel.log()
            # just in case errors are non-fatal
            raster = None

        # attach the raster
        self.raster = raster

        # all done
        return self


    def setSpec(self, origin, shape):
        """
        Reconfigure my {spec}
        """
        # save the {origin} and {shape}
        self.origin = origin
        self.shape = shape
        # make a spec
        spec = self.Spec(origin=origin, shape=shape)
        # attach it
        self.spec = spec
        # and return it
        return spec


    # metamethods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # load my product spec
        self.spec = self.Spec(origin=self.origin, shape=self.shape)
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
        yield f"{margin}name: {self.pyre_name}"
        yield f"{margin}family: {self.pyre_family()}"
        yield f"{margin}data: {self.data}"
        yield f"{margin}shape: {self.shape}"
        yield f"{margin}pixels: {self.cells()}"
        yield f"{margin}footprint: {self.bytes()} bytes"
        # all done
        return


# end of file
