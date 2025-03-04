# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2025 all rights reserved
#


# the package
import ampcor


# ampcor is a workflow
class Ampcor(ampcor.flow.flow, family="ampcor.workflows"):
    """
    The workflow that produces an offset map from an SLC to another
    """


    # framework hooks
    @classmethod
    def pyre_default(cls, **kwds):
        """
        Provide access to the default implementation of this flow
        """
        # invoke the foundry to get the default implementation and publish it
        return ampcor.workflows.ampcor()


# end of file
