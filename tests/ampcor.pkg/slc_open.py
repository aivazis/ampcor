#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2022 all rights reserved
#


"""
Sanity check: verify that the {ampcor} package is accessible
"""


def test():
    # access the {ampcor} package
    import ampcor
    # and the journal
    import journal

    # make a channel
    channel = journal.debug("ampcor.slc")

    # activate some channels
    # channel.activate()
    # journal.debug("pyre.memory.direct").activate()

    # create an SLC
    slc = ampcor.products.newSLC(name="ref")

    # verify the configuration
    assert slc.shape == (36864, 10344)
    assert slc.data == ampcor.primitives.path("../../data/20061231.slc")

    # show me
    channel.line(f"slc:")
    for line in slc.show(indent=" "*2, margin=" "*2):
        channel.line(line)
    channel.log()

    # verify the raster is trivial before opening the payload
    assert slc.raster is None
    # load some real data
    slc.open()
    # verify the raster is now non-trivial
    assert slc.raster is not None
    # and in fact it is an instance of the slc raster object from {libampcor}
    assert isinstance(slc.raster, ampcor.libampcor.SLCConstRaster)

    # all done
    return 0


# main
if __name__ == "__main__":
    # do...
    status = test()
    # share
    raise SystemExit(status)


# end of file
