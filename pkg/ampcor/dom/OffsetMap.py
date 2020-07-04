# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# framework
import ampcor


# declaration
class OffsetMap:
    """
    A logically Cartesian map that establishes a correspondence between a collection of points
    on a {reference} raster and a {secondary} raster
    """


    # public data
    @property
    def size(self):
        """
        Compute the total number of elements in the map
        """
        # easy enough
        return self.tile.size


    @property
    def shape(self):
        """
        Return the shape of the map
        """
        # easy enough
        return self.tile.shape


    @property
    def layout(self):
        """
        Return the index packing order
        """
        # easy enough
        return self.tile.layout


    # meta-methods
    def __init__(self, shape, layout=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # storage
        self.domain = []
        self.codomain = []
        # access as a Cartesian map
        self.tile = ampcor.grid.tile(shape=shape, layout=layout)
        # all done
        return


    def __getitem__(self, index):
        """
        Return the pair of correlated points stored at {index}
        """
        # ask my tile for the offset
        offset = self.tile.offset(index)
        # pull the corresponding points
        ref = self.domain[offset]
        sec = self.codomain[offset]
        # and return them
        return (ref, sec)


    def __setitem__(self, index, points):
        """
        Return the value stored at {index}
        """
        # ask my tile for the offset
        offset = self.tile.offset(index)
        # unpack the points
        ref, sec = points
        # store them
        self.domain[offset] = ref
        self.codomain[offset] = sec
        # all done
        return


    def __len__(self):
        """
        Compute my length
        """
        # delegate to the corresponding property
        return self.size


# end of file
