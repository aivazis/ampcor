# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# support
import ampcor


# application engine
class Panel(ampcor.shells.command, family="ampcor.cli.ux"):
    """
    Select application behavior that is mapped to the capabilities of the web client
    """


    # my workflow
    flow = ampcor.specs.ampcor()
    flow.doc = "the ampcor workflow"


    # metamethods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # show me my flow
        print(self.flow)
        # all done
        return


# end of file
