# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2022 all rights reserved
#


# externals
import ampcor
import journal


# declaration
class Plan(ampcor.shells.command, family='ampcor.cli.plan'):
    """
    Examine the proposed correlation plan
    """


    # my workflow
    flow = ampcor.specs.ampcor()
    flow.doc = "the ampcor workflow"


    # commands
    @ampcor.export(tip="describe the plan")
    def info(self, plexus, **kwds):
        """
        Describe the proposed correlation plan
        """
        # make a channel
        channel = journal.info("ampcor.plan.map")

        # show everything
        correlator = self.flow.correlator
        # generate a report
        report = correlator.show(margin="", indent="  ")
        # show me
        channel.report(report)

        # and flush
        channel.log()

        # all done
        return 0


# end of file
