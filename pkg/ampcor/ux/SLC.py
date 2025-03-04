# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2025 all rights reserved
#


# wrapper over the input data products that helps with viz
class SLC:


    # public data
    @property
    def range(self):
        """
        Return the range of values in the data product
        """
        # get the cached values
        range = self._range
        # if they are good
        if range is not None:
            # use them
            return range
        # otherwise, compute
        range = self._slc.raster.range
        # save
        self._range = range
        # and return
        return range


    @property
    def raster(self):
        """
        Access the SLC raster
        """
        # easy enough
        return self._slc.raster


    # metamethods
    def __init__(self, slc, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the product
        self._slc = slc.open(mode="r")
        # all done
        return


    # data
    _slc = None
    _range = None


# end of file
