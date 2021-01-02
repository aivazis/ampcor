#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2021 all rights reserved
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

    # get the layout
    layout = slc.layout

    # make sure we get the same information from the {layout} as from the spec
    assert slc.cells() == layout.cells

    # all done
    return 0


# main
if __name__ == "__main__":
    # do...
    status = test()
    # share
    raise SystemExit(status)


# end of file
