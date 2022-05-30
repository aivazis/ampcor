# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2022 all rights reserved
#


# framework
import ampcor


# declaration
class Plan:
    """
    Encapsulation of the computational work necessary to refine an offset map between a
    {reference} and a {secondary} image
    """


    # public data
    # known at construction
    tile = None      # my shape as a grid of correlation pairs
    chip = None      # the shape of the reference chip
    padding = None   # the padding to apply to form the search window in the secondary raster
    # deduced
    window = None    # the window in the secondary raster
    reference = None # the sequence of reference tiles
    secondary = None # the sequence of secondary search windows


    @property
    def cells(self):
        """
        Compute the total number of cells required to store the reference and secondary tiles
        """
        # get the number of pairings
        pairs = len(self.map)

        # the total number of cells needed to store the reference tiles
        ref = pairs * self.chip.cells
        # and the total number of cells needed to store the secondary tiles
        sec = pairs * self.window.cells

        # all done
        return ref, sec


    # meta-methods
    def __init__(self, correlator, map, **kwds):
        # chain up
        super().__init__(**kwds)

        # save the pairings
        self.map = map
        # make me a tile so i can behave like a grid
        self.tile = ampcor.grid.tile(shape=correlator.offsets.shape)

        # get the grid bindings
        libgrid = ampcor.libpyre.grid

        # get the reference tile size
        self.chip = libgrid.Shape2D(shape=correlator.chip)
        # and the search window padding
        self.padding = libgrid.Shape2D(shape=correlator.padding)
        # compute the secondary window shape
        self.window = self.chip + 2 * self.padding

        # all done
        return


    def __len__(self):
        """
        By definition, my length is the number of tile pairs
        """
        # easy enough
        return len(self.map)


    # debugging
    def show(self, indent, margin):
        """
        Display details about this plan
        """
        slc = ampcor.products.slc()
        # so we can ask it for its size
        slcPixel = slc.bytesPerPixel

        # sign on
        yield f"{margin}plan:"
        # tile info
        yield f"{margin}{indent}pairs: {len(self)}"
        yield f"{margin}{indent}shape: {self.tile.shape}, layout: {self.tile.layout}"
        # memory footprint
        refCells, secCells = self.cells
        refBytes = refCells * slcPixel / 1024**3
        secBytes = secCells * slcPixel / 1024**3
        yield f"{margin}{indent}arena footprint:"
        yield f"{margin}{indent*2}reference: {refCells} cells in {refBytes:.3f} Gb"
        yield f"{margin}{indent*2}secondary: {secCells} cells in {secBytes:.3f} Gb"
        yield f"{margin}{indent*2}    total: {refBytes + secBytes:.3f} Gb"

        # don't show the actual points; there may be too many of them
        return


# end of file
