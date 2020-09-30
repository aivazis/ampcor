# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# externals
import graphql
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

        # show me, for now
        plexus.info.line(f"operation: {operation}")
        plexus.info.line(f"{query}")
        plexus.info.line(f"variables: {variables}")
        plexus.info.log()

        # just return the version, for now
        meta = ampcor.meta
        # build the response
        doc = {
            "data": {
                "version": {
                    "major": meta.major,
                    "minor": meta.minor,
                    "micro": meta.micro,
                    "revision": meta.revision,
                },

                "flow": {
                    "id": "client:root:flow",
                    "__typename": "Ampcor",
                    "name": "LA",
                    "family": "ampcor.workflows.ampcor",

                    "reference": {
                        "id": "client:root:flow:reference",
                        # "__typename": "SLC",
                        "name": "20061231",
                        "family": "ampcor.products.slc.slc",
                        "shape": [36864, 10344],
                        "exists": True,
                    },

                    "secondary": {
                        "id": "client:root:flow:secondary",
                        # "__typename": "SLC",
                        "name": "20070215",
                        "family": "ampcor.products.slc.slc",
                        "shape": [36864, 10344],
                        "exists": True,
                    },
                },
            },
        }
        # and hand it to the client as a {json} document
        return server.documents.JSON(server=server, value=doc)

        # execute the query and get the response
        response = graphql.graphql(self.schema, query, None, request, variables, operation)

        # build the result
        doc = { "data": response.data }
        # in addition, if something went wrong
        if response.errors:
            # inform the client
            doc["errors"] = [ {"message": error.message} for error in response.errors ]

        # turn it into a document and serve it
        return server.documents.JSON(server=server, value=doc)


    # metamethods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)

        # build my schema
        self.schema = None

        # all done
        return


# end of file
