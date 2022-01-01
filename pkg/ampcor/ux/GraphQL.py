# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2022 all rights reserved
#


# externals
import graphene
import json
# support
import ampcor


# the {graphql} request handler
class GraphQL:


    # interface
    def respond(self, plexus, server, request, **kwds):
        """
        Resolve the {query} and generate a response for the client
        """
        # parse the {request} payload
        payload = json.loads(b'\n'.join(request.payload))
        # and unpack it
        query = payload.get("query")
        operation = payload.get("operation")
        variables = payload.get("variables")

        # execute the query
        result = self.schema.execute(query, context=self.context)

        # assemble the resulting document
        doc = { "data": result.data }
        # in addition, if something went wrong
        if result.errors:
            # inform the client
            doc["errors"] = [ {"message": error.message} for error in result.errors ]

        # encode it using JSON and serve it
        return server.documents.JSON(server=server, value=doc)


    # metamethods
    def __init__(self, panel, **kwds):
        # chain up
        super().__init__(**kwds)

        # load the schema
        from .schema import schema
        # build my schema
        self.schema = schema

        # set up the execution context
        self.context = {
            "flow": panel.flow,
            "version": ampcor.meta,
        }

        # all done
        return


# end of file
