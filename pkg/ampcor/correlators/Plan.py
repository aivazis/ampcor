# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
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
    def pairs(self):
        """
        Yield valid pairs of reference and secondary tiles
        """
        # go through my tile containers
        for ref, sec in zip(self.reference, self.secondary):
            # invariant: either both are good, or both are bad
            if ref and sec:
                # yield them
                yield ref, sec
        # all done
        return


    @property
    def bytes(self):
        """
        Compute the total amount of memory required to store the reference and secondary tiles
        """
        # the reference footprint
        ref = sum(tile.bytes for tile in filter(None, self.reference))
        # the secondary footprint
        sec = sum(tile.bytes for tile in filter(None, self.secondary))
        # all done
        return ref, sec


    # meta-methods
    def __init__(self, correlator, regmap, rasters, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my tile
        self.tile = regmap.tile

        # get the reference tile size
        self.chip = correlator.chip
        # and the search window padding
        self.padding = correlator.padding
        # compute the secondary window shape
        self.window = tuple(c+2*p for c,p in zip(self.chip, self.padding))

        # initialize my containers
        self.reference, self.secondary = self.assemble(regmap=regmap, rasters=rasters)

        # all done
        return


    def __len__(self):
        """
        By definition, my length is the number of valid tile pairs
        """
        # invariant: either both tiles are good, or both are bad
        return len(tuple(filter(None, self.reference)))


    def __getitem__(self, index):
        """
        Behave like a grid
        """
        # ask my shape tile to resolve the index
        offset = self.tile.offset(index)
        # grab the corresponding tiles
        ref = self.reference[offset]
        sec = self.secondary[offset]
        # and return them
        return ref, sec


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

        # initialize the tile containers
        referenceTiles = []
        secondaryTiles = []

        # go through matching pairs of points in the initial guess
        for ref, sec in zip(regmap.domain, regmap.codomain):
            # form the upper left hand corner of the reference tile
            origin = tuple(r - c//2 for r,c in zip(ref, chip))
            # attempt to make a slice; invalid specs get rejected by the slice factory
            refSlice = reference.slice(origin=origin, shape=chip)

            # the upper left hand corner of the secondary tile
            origin = tuple(t - c//2 - p for t,c,p in zip(sec, chip, padding))
            # and its shape
            shape = tuple(c + 2*p for c,p in zip(chip, padding))
            # try to turn this into a slice
            secSlice = secondary.slice(origin=origin, shape=shape)

            # if both slices are valid
            if refSlice and secSlice:
                # push them into their respective containers
                referenceTiles.append(refSlice)
                secondaryTiles.append(secSlice)
            # otherwise
            else:
                # push invalid slices for both of them
                referenceTiles.append(None)
                secondaryTiles.append(None)

        # all done
        return referenceTiles, secondaryTiles


    # interface
    def show(self, indent, margin):
        """
        Display details about this plan
        """
        # sign on
        yield f"{margin}plan:"
        # tile info
        yield f"{margin}{indent}shape: {self.tile.shape}, layout: {self.tile.layout}"
        yield f"{margin}{indent}pairs: {len(self)} out of {self.tile.size}"
        # memory footprint
        refBytes, secBytes = self.bytes
        yield f"{margin}{indent}footprint:"
        yield f"{margin}{indent*2}reference: {refBytes} bytes"
        yield f"{margin}{indent*2}secondary: {secBytes} bytes"

        # go through the pairs
        for offset, (ref,sec) in enumerate(zip(self.reference, self.secondary)):
            # compute the index of this pair
            index = self.tile.index(offset)
            # if this is a valid pair
            if ref and sec:
                # identify the pair
                yield f"{margin}{indent}pair: {index}"
                # show me the reference slice
                yield f"{margin}{indent*2}ref:"
                yield from ref.show(indent, margin=margin+3*indent)
                # and the secondary slice
                yield f"{margin}{indent*2}sec:"
                yield from sec.show(indent, margin=margin+3*indent)
            # otherwise
            else:
                # identify the pair as invalid
                yield f"{margin}{indent}pair: {index} INVALID"

        # all done
        return


# end of file
