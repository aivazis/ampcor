# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# access the pyre framework
import pyre
# and my package
import ampcor


# declaration
class Ampcor(pyre.plexus, family='ampcor.shells.plexus'):
    """
    The main action dispatcher
    """

    # types
    from .Action import Action as pyre_action


    # pyre framework hooks
    # support for the help system
    def pyre_banner(self):
        """
        Generate the help banner
        """
        # show the license header
        return ampcor.meta.license


    # interactive session management
    def pyre_interactiveSessionContext(self, context=None):
        """
        Go interactive
        """
        # prime the execution context
        context = context or {}
        # grant access to my package
        context['ampcor'] = ampcor  # my package
        # and chain up
        return super().pyre_interactiveSessionContext(context=context)


# end of file
