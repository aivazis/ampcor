# -*- Python -*-
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
    def footprint(self):
        """
        Compute the total amount of memory required to store the reference and secondary tiles
        """
        # the reference footprint
        ref = sum(tile.footprint for tile in filter(None, self.reference))
        # the secondary footprint
        sec = sum(tile.footprint for tile in filter(None, self.secondary))
        # all done
        return ref, sec


    # interface
    def show(self, channel):
        """
        Display details about this plan in {channel}
        """
        # sign on
        channel.line(f" -- plan:")
        # tile info
        channel.line(f"        shape: {self.tile.shape}, layout: {self.tile.layout}")
        channel.line(f"        pairs: {len(self)} out of {self.tile.size}")
        # memory footprint
        refFootprint, secFootprint = self.footprint
        channel.line(f"        footprint:")
        channel.line(f"            reference: {refFootprint} bytes")
        channel.line(f"            secondary: {secFootprint} bytes")

        # go through the pairs
        for offset, (ref,sec) in enumerate(zip(self.reference, self.secondary)):
            # compute the index of this pair
            index = self.tile.index(offset)
            # if this is a valid pair
            if ref and sec:
                # identify the pair
                channel.line(f"        pair: {index}")
                # show me the reference slice
                channel.line(f"            ref:")
                ref.show(channel)
                # and the secondary slice
                channel.line(f"            sec:")
                sec.show(channel)
            # otherwise
            else:
                # identify the pair as invalid
                channel.line(f"        pair: {index} INVALID")

        # all done
        return


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
            begin = tuple(r - c//2 for r,c in zip(ref, chip))
            # attempt to make a slice; invalid specs get rejected by the slice factory
            refSlice = reference.slice(begin=begin, shape=chip)

            # the upper left hand corner of the secondary tile
            begin = tuple(t - c//2 - p for t,c,p in zip(sec, chip, padding))
            # and its shape
            shape = tuple(c + 2*p for c,p in zip(chip, padding))
            # try to turn this into a slice
            secSlice = secondary.slice(begin=begin, shape=shape)

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


# end of file
