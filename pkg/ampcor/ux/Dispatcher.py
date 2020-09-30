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

# the query handler
from .GraphQL import GraphQL


# the main request dispatcher
class Dispatcher:
    """
    The handler of web requests
    """


    # interface
    def dispatch(self, plexus, server, request):
        """
        Analyze the {request} received by the {server} and invoke the appropriate {plexus} behavior
        """
        # get the request type
        command = request.command
        # get the request uri
        url = request.url
        # show me
        plexus.info.log(f"{command:>4}: {url}")
        # take a look
        match = self.regex.match(url)
        # if there is no match
        if not match:
            # something terrible has happened
            return server.responses.NotFound(server=server)

        # find who matched
        token = match.lastgroup
        # look up the handler
        handler = getattr(self, token)
        # invoke
        return handler(plexus=plexus, server=server, request=request, match=match)


    # metamethods
    def __init__(self, plexus, docroot, pfs, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the location of my document root so i can serve static assets
        self.docroot = docroot.discover()
        # attach it to the app's private filesystem
        pfs['ux'] = docroot

        # make a spec for the app engine
        spec = 'ux'
        # use it to build a name for it
        name = f"{plexus.pyre_name}.{spec}"
        # make an instance of the application engine
        self.panel = ampcor.ux.panel(name=name, spec=spec, plexus=plexus, globalAliases=True)

        # instantiate the {GraphQL} handler
        self.gql = GraphQL()

        # all done
        return


    # handlers
    def tile(self, plexus, server, request, **kwds):
        """
        Build and serve a bitmap of a coarse amplitude ref tile
        """
        uri = "/ux/uni1d.bmp"
        # send a bitmap to the client
        return server.documents.File(uri=uri, server=server, application=plexus)


    def ref(self, **kwds):
        """
        Build and serve a tile from the {ref} SLC input data product
        """
        # ask the panel
        return self.panel.refTile(**kwds)


    def rac(self, plexus, server, request, **kwds):
        """
        Build and serve a bitmap of a coarse amplitude ref tile
        """
        # tell me
        plexus.info.log(" rac: preparing raster")
        # ask the plexus for its {mdy} panel
        mdy = plexus.pyre_repertoir.resolve(plexus=plexus, spec="mdy")
        # ask it to build me a bitmap
        data = mdy.rac(plexus=plexus)
        # and hand it to the client
        return server.documents.BMP(server=server, value=data)


    def graphql(self, **kwds):
        """
        Handle a {graphql} request
        """
        # delegate to my {graphql} handler
        return self.gql.respond(**kwds)


    def stop(self, plexus, server, **kwds):
        """
        The client is asking me to die
        """
        # log it
        plexus.info.log("shutting down")
        # and exit
        return server.documents.Exit(server=server)


    def document(self, plexus, server, request, **kwds):
        """
        The client requested a document from the {plexus} pfs
        """
        # form the uri
        uri = "/ux" + request.url
        # open the document and serve it
        return server.documents.File(uri=uri, server=server, application=plexus)


    def favicon(self, plexus, server, request, **kwds):
        """
        The client requested the app icon
        """
        # we don't have one
        return server.responses.NotFound(server=server)


    def root(self, plexus, server, request, **kwds):
        """
        The client requested the root document
        """
        # form the uri
        uri = "/ux/{0.pyre_namespace}.html".format(plexus)
        # open the document and serve it
        return server.documents.File(uri=uri, server=server, application=plexus)


    # private data
    regex = re.compile("|".join([
        r"/(?P<tile>exp/tile)",
        r"/(?P<ref>slc/ref/tile-(?P<refTileZoom>[0-9])@(?P<refTileOrigin>[0-9]+x[0-9]+)\+(?P<refTileShape>[0-9]+x[0-9]+))",
        r"/(?P<rac>ref/amplitude/coarse)",
        r"/(?P<graphql>graphql)",
        r"/(?P<stop>stop)",
        r"/(?P<document>(graphics/.+)|(styles/.+)|(fonts/.+)|(.+\.js))",
        r"/(?P<favicon>favicon.ico)",
        r"/(?P<root>.*)",
        ]))


# end of file
