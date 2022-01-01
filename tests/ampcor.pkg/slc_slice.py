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
    # journal.debug("pyre.memory.direct").activate()
    # journal.debug("ampcor.slc").activate()

    # create an SLC
    slc = ampcor.products.newSLC(name="ref")

    # pick an origin
    origin = 0,0
    # and a shape
    shape = 128, 128
    # make a slice
    slice = slc.slice(origin=origin, shape=shape)

    # check
    assert tuple(slice.origin) == origin
    assert tuple(slice.shape) == shape

    # all done
    return 0


# main
if __name__ == "__main__":
    # do...
    status = test()
    # share
    raise SystemExit(status)


# end of file
