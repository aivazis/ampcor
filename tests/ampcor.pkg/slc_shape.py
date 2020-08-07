#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
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

    # show me
    channel.line(f"slc:")
    for line in slc.show(indent=" "*2, margin=" "*2):
        channel.line(line)
    channel.log()

    # verify the configuration
    assert slc.shape == (36864, 10344)
    assert slc.data == ampcor.primitives.path("../../data/20061231.slc")

    # verify it has a spec
    assert slc.spec
    # and that it responds correctly to size queries
    assert slc.cells() == 36864 * 10344
    assert slc.bytes() == slc.cells() * 2*4

    # all done
    return 0


# main
if __name__ == "__main__":
    # do...
    status = test()
    # share
    raise SystemExit(status)


# end of file
