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


# the main request dispatcher
class UX:
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
        plexus.info.log(f"{command}: {url}")
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
    def __init__(self, docroot, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the location of my document root so i can serve static assets
        self.docroot = docroot
        # all done
        return


    # handlers
    def version(self, plexus, server, **kwds):
        """
        The client requested the version of the server
        """
        # get the package metadata
        meta = ampcor.meta
        # build a version tag
        vtag = f"{meta.major}.{meta.minor}.{meta.micro} rev {meta.revision}"
        # put it together
        tag = f"{plexus.pyre_namespace} version {vtag}, built on {meta.date}"
        # and hand it to the client
        return server.documents.JSON(server=server, value=tag)


    def stop(self, plexus, **kwds):
        """
        The client is asking me to die
        """
        # log it
        plexus.info.log("shutting down")
        # and exit
        raise SystemExit(0)


    def document(self, plexus, server, request, **kwds):
        """
        The client requested a document from the {plexus} pfs
        """
        # form the uri
        uri = "/ux" + request.url
        # open the document and serve it
        return server.documents.File(uri=uri, server=server, application=plexus)


    def root(self, plexus, server, request, **kwds):
        """
        The client requested the root document
        """
        # let me know
        plexus.info.log("serving the root document")
        # form the uri
        uri = "/ux/{0.pyre_namespace}.html".format(plexus)
        # open the document and serve it
        return server.documents.File(uri=uri, server=server, application=plexus)


    # private data
    regex = re.compile("|".join([
        r"/(?P<version>query/meta/version)",
        r"/(?P<stop>action/meta/stop)",
        r"/(?P<document>(graphics/.+)|(styles/.+)|(fonts/.+)|(.+\.js))",
        r"/(?P<root>.*)",
        ]))


# end of file
