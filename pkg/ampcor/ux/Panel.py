# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# support
import ampcor


# local types
from .SLC import SLC


# application engine
class Panel(ampcor.shells.command, family="ampcor.cli.ux"):
    """
    Select application behavior that is mapped to the capabilities of the web client
    """


    # my workflow
    flow = ampcor.specs.ampcor()
    flow.doc = "the ampcor workflow"


    # public data
    @property
    def ref(self):
        """
        Access to the raw reference SLC input data product
        """
        # get my cache
        ref = self._ref
        # if it's valid
        if ref:
            # all done
            return ref

        # otherwise, get the reference raster from the flow and wrap it
        ref = SLC(slc=self.flow.reference)
        # cache it
        self._ref = ref
        # and return it
        return ref


    # interface
    def refTile(self, plexus, server, request, match, **kwds):
        # get the tile
        tile = int(match["refTile"])
        # its width
        width = int(match["refTileWidth"])
        # and height
        height = int(match["refTileHeight"])
        # and the zoom level
        zoom = int(match["refzoom"])
        # get my {ref} wrapper
        ref = self.ref
        # ask it for the value range
        range = (0, 100) # or use {ref.range}, if you don't mind waiting a bit...
        # build the bitmap
        bitmap = ampcor.libampcor.viz.slc(raster=ref.raster,
                                          tile=tile, shape=(height,width), zoom=zoom,
                                          range=range)
        # and respond
        return server.documents.BMP(server=server, bmp=memoryview(bitmap))


    # implementation details
    # input data products
    _ref = None   # the reference SLC
    _sec = None   # the secondary SLC raster
    # the output data product
    _offsets = None     # the offset map

    # arenas
    # reference amplitude arenas
    _coarseRef = None
    _refinedRef = None
    # secondary amplitude arenas
    _coarseSec = None
    _refinedSec = None

    # correlation surfaces
    _coarseGamma = None
    _refinedGamma = None
    _zoomedGamma = None
    _zoomedComplexGamma = None

# end of file
