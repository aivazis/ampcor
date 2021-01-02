#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2021 all rights reserved
#


"""
Create an {offsets} product
"""


# externals
import itertools


# the driver
def test():
    # access the {ampcor} package
    import ampcor
    # create a map
    offsets = ampcor.products.newOffsets(name="offsets")
    # open its raster
    offsets.open(mode="r")

    # visit all the spots
    for idx in itertools.product(*map(range, offsets.shape)):
        # get the pixel
        p = offsets[idx]
        # check the reference location
        assert p.ref == (128*idx[0], 128*idx[1])
        assert p.delta == idx
        assert p.confidence == 1
        assert p.snr == 0.5
        assert p.covariance == 1

    # all done
    return 0


# main
if __name__ == "__main__":
    # do...
    status = test()
    # share
    raise SystemExit(status)


# end of file
