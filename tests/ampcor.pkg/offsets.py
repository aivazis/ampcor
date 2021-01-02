#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2021 all rights reserved
#


"""
Sanity check: verify that the {offsets} factory is accessible
"""


def test():
    # access the {ampcor} package
    import ampcor
    # create a map
    offsets = ampcor.products.newOffsets(name="offsets")
    # all done
    return 0


# main
if __name__ == "__main__":
    # do...
    status = test()
    # share
    raise SystemExit(status)


# end of file
