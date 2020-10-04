# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# externals
import re
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


    @property
    def sec(self):
        """
        Access to the raw secondary SLC input data product
        """
        # get my cache
        sec = self._sec
        # if it's valid
        if sec:
            # all done
            return sec

        # otherwise, get the secondary raster from the flow and wrap it
        sec = SLC(slc=self.flow.secondary)
        # cache it
        self._sec = sec
        # and return it
        return sec


    # interface
    def slc(self, plexus, server, request, match, **kwds):
        """
        Render a tile from one of the input SLC data products
        """
        # parse the {request} uri
        match = self.slcRegex.match(request.url)
        # if something is wrong
        if match is None:
            # we have a bug; get a channel
            channel = plexus.firewall
            # and complain
            channel.line(f"while scanning '{request.url}'")
            channel.log(f"couldn't understand the URI")
            # if firewalls aren't fatal
            return server.documents.NorFound(server=server)

        # extract the slc
        slc = match["slc"]
        # the zoom level
        zoom = int(match["zoom"])
        # the signal
        signal = match["signal"]
        # the origin
        origin = tuple(map(int, match["origin"].split("x")))
        # and the shape
        shape = tuple(map(int, match["shape"].split("x")))

        # get the data product
        product = getattr(self, slc)

        # if we are showing amplitude
        if signal == "amplitude":
            # set the value range
            range = (0, 1000) # or use {product.range}, if you don't mind waiting a bit...
            # get the bitmap factory
            viz = ampcor.libampcor.viz.slcAmplitude
            # build the bitmap
            bitmap = viz(raster=product.raster,
                         origin=origin, shape=shape, zoom=zoom,
                         range=range)
            # and respond
            return server.documents.BMP(server=server, bmp=memoryview(bitmap))

        # if we are showing phase
        if signal == "phase":
            # get the bitmap factory
            viz = ampcor.libampcor.viz.slcPhase
            # build the bitmap
            bitmap = viz(raster=product.raster,
                         origin=origin, shape=shape, zoom=zoom)
            # and respond
            return server.documents.BMP(server=server, bmp=memoryview(bitmap))

        # if we are showing the full complex value
        if signal == "complex":
            # set the value range
            range = (0, 1000) # or use {ref.range}, if you don't mind waiting a bit...
            # get the bitmap factory
            viz = ampcor.libampcor.viz.slc
            # build the bitmap
            bitmap = viz(raster=product.raster,
                         origin=origin, shape=shape, zoom=zoom,
                         range=range)
            # and respond
            return server.documents.BMP(server=server, bmp=memoryview(bitmap))

        # otherwise, we have a bug; get a channel
        channel = plexus.firewall
        # and complain
        channel.log(f"while scanning {request.url}")
        channel.log(f"couldn't understand the signal type")
        # if firewalls aren't fatal
        return server.documents.NorFound(server=server)


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

    # the SLC request parser
    slcRegex = re.compile("".join([
        r"/slc/",
        r"(?P<slc>(ref)|(sec))",
        r"/",
        r"(?P<signal>(amplitude)|(phase)|(complex))",
        r"/",
        r"(?P<zoom>-?[0-9]+)",
        r"/",
        r"(?P<shape>[0-9]+x[0-9]+)",
        r"/",
        r"(?P<origin>[0-9]+x[0-9]+)",
        r"$",
    ]))



# end of file
