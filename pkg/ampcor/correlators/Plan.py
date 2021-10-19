# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2021 all rights reserved
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
        # initialize the footprints
        ref = 0
        sec = 0
        # go through the tiles
        for _, refTile, secTile in self.tiles:
            # compute their footprints and update the counters
            ref += refTile.cells
            sec += secTile.cells
        # all done
        return ref, sec


    # meta-methods
    def __init__(self, correlator, regmap, rasters, **kwds):
        # chain up
        super().__init__(**kwds)

        # get the reference tile size
        self.chip = correlator.chip
        # and the search window padding
        self.padding = correlator.padding
        # make me a tile so i can behave like a grid
        self.tile = ampcor.grid.tile(shape=correlator.offsets.shape)

        # compute the secondary window shape
        self.window = tuple(c+2*p for c,p in zip(self.chip, self.padding))

        # initialize my container
        self.tiles = tuple(self.assemble(regmap=regmap, rasters=rasters))

        # all done
        return


    def __len__(self):
        """
        By definition, my length is the number of valid tile pairs
        """
        # invariant: either both tiles are good, or both are bad
        return len(self.tiles)


    # implementation details
    def assemble(self, rasters, regmap):
        """
        Form the set of pairs of tiles to correlate in order to refine {regmap}, a coarse offset
        map from a reference image to a secondary image
        """
        # unpack the rasters
        reference, secondary = rasters
        # get the reference tile size
        chip = self.chip
        # and the search window padding
        padding = self.padding

        # go through matching pairs of points in the initial guess
        for pid, (ref, sec) in enumerate(zip(*regmap)):
            # form the upper left hand corner of the reference tile
            origin = tuple(r - c//2 for r,c in zip(ref, chip))
            # attempt to make a slice; invalid specs get rejected by the slice factory
            refSlice = reference.slice(origin=origin, shape=chip)
            # if the slice is not a good one
            if not refSlice:
                # move on
                continue

            # the upper left hand corner of the secondary tile
            origin = tuple(s - c//2 - p for s,c,p in zip(sec, chip, padding))
            # and its shape
            shape = tuple(c + 2*p for c,p in zip(chip, padding))
            # try to turn this into a slice
            secSlice = secondary.slice(origin=origin, shape=shape)
            # if either slice is invalid
            if not secSlice:
                # move on
                continue

            # if both are good, mark them and publish them
            yield pid, refSlice, secSlice

        # all done
        return


    # interface
    def show(self, indent, margin):
        """
        Display details about this plan
        """
        # get the slc product spec
        slc = ampcor.products.slc()
        # so we can ask it for its size
        slcPixel = slc.bytesPerPixel

        # sign on
        yield f"{margin}plan:"
        # tile info
        yield f"{margin}{indent}shape: {self.tile.shape}, layout: {self.tile.layout}"
        yield f"{margin}{indent}pairs: {len(self)} out of {self.tile.size}"
        # memory footprint
        refCells, secCells = self.cells
        refBytes = refCells * 8
        secBytes = secCells * 8
        yield f"{margin}{indent}arena footprint:"
        yield f"{margin}{indent*2}reference: {refCells} cells in {refBytes} bytes"
        yield f"{margin}{indent*2}secondary: {secCells} cells in {secBytes} bytes"
        yield f"{margin}{indent*2}    total: {refBytes + secBytes} bytes"

        return

        # go through the pairs
        for offset, ref,sec in self.tiles:
            # compute the index of this pair
            index = self.tile.index(offset)
            # if this is a valid pair
            if ref and sec:
                # identify the pair
                yield f"{margin}{indent}pair: {index}"
                # show me the reference slice
                yield f"{margin}{indent*2}ref:"
                yield f"{margin}{indent*3}origin: {tuple(ref.origin)}"
                yield f"{margin}{indent*3}shape: {tuple(ref.shape)}"
                # and the secondary slice
                yield f"{margin}{indent*2}sec:"
                yield f"{margin}{indent*3}origin: {tuple(sec.origin)}"
                yield f"{margin}{indent*3}shape: {tuple(sec.shape)}"
            # otherwise
            else:
                # identify the pair as invalid
                yield f"{margin}{indent}pair: {index} INVALID"

        # all done
        return


# end of file
