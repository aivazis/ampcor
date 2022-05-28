# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2022 all rights reserved
#


# externals
import ampcor


# declaration
class Plan(ampcor.shells.command, family='ampcor.cli.plan'):
    """
    Examine the proposed correlation plan
    """


    @ampcor.export(tip="describe the plan")
    def info(self, plexus, **kwds):
        """
        Describe the proposed correlation plan
        """
        # all done
        return 0


# end of file
