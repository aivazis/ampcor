#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# externals
import itertools
import pyre
import ampcor


# the app
class Dump(pyre.application):
    """
    Dump the contents of the {offsets} product
    """


    # user configurable state
    flow = pyre.properties.str()
    flow.default = "LA"
    flow.doc = "the configuration to load"

    map = pyre.properties.str()
    map.default = "single"
    map.doc = "the name of the map to dump"


    # obligations
    @ampcor.export
    def main(self, *args, **kwds):
        """
        The main entry point
        """
        # load the workflow
        pyre.loadConfiguration(f"{self.flow}.pfg")

        # create a map
        offsets = ampcor.products.newOffsets(name=self.map)
        # open its raster
        offsets.open(mode="r")

        # visit all the spots
        for idx in itertools.product(*map(range, offsets.shape)):
            # get the pixel
            p = offsets[idx]
            # show me
            print(f"map[{idx}]:")
            print(f"  ref: {p.ref}")
            print(f"  delta: {p.delta}")
            print(f"  confidence: {p.confidence}")
            print(f"  snr: {p.snr}")
            print(f"  covariance: {p.covariance}")

        # all done
        return 0


# main
if __name__ == "__main__":
    # instantiate
    app = Dump(name="dump")
    # invoke
    status = app.run()
    # share
    raise SystemExit(status)


# end of file
