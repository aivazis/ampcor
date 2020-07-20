# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# externals
import ampcor


# declaration
class Offsets(ampcor.shells.command, family="ampcor.cli.offsets"):
    """
    Estimate an offset field given a pair of rasters
    """


    # user configurable state
    reference = ampcor.dom.raster()
    reference.doc = "the reference raster image"

    secondary = ampcor.dom.raster()
    secondary.doc = "the secondary raster image"

    correlator = ampcor.correlators.correlator()
    correlator.doc = "the calculator of the offset field"


    # behaviors
    @ampcor.export(tip="produce the offset field between the {reference} and {secondary} rasters")
    def estimate(self, plexus, **kwds):
        """
        Produce an offset map between the {reference} and {secondary} images
        """
        # get the reference image
        reference = self.reference
        # the secondary image
        secondary = self.secondary
        # and the correlator
        correlator = self.correlator
        # ask the correlator to do its thing
        return correlator.estimate(plexus=plexus, reference=reference, secondary=secondary, **kwds)


    @ampcor.export(tip="display my configuration")
    def info(self, plexus, **kwds):
        """
        Display the action configuration
        """
        # grab a channel
        channel = plexus.info

        # get things going
        channel.line()

        # shell
        channel.line(f" -- shell: {plexus.shell}")
        channel.line(f"    hosts: {plexus.shell.hosts}")
        channel.line(f"    tasks: {plexus.shell.tasks} per host")
        channel.line(f"    gpus:  {plexus.shell.gpus} per task")

        # inputs
        channel.line(f" -- data files")
        # reference raster
        channel.line(f"    reference: {self.reference}")
        if self.reference:
            channel.line(f"        data: {self.reference.data}")
            channel.line(f"        shape: {self.reference.shape}")
            channel.line(f"        pixels: {self.reference.cells()}")
            channel.line(f"        footprint: {self.reference.bytes()} bytes")
        # secondary raster
        channel.line(f"    secondary: {self.secondary}")
        if self.secondary:
            channel.line(f"        data: {self.secondary.data}")
            channel.line(f"        shape: {self.secondary.shape}")
            channel.line(f"        pixels: {self.secondary.cells()}")
            channel.line(f"        footprint: {self.secondary.bytes()} bytes")

        # unpack my arguments
        reference = self.reference
        secondary = self.secondary
        correlator = self.correlator

        # show the correlator configuration
        correlator.show(channel=channel)
        # ask the correlator for the coarse map
        coarse = correlator.coarse.map(reference=reference)
        # make a plan
        plan = correlator.makePlan(regmap=coarse, rasters=(reference, secondary))
        # and show me the plan details
        plan.show(channel=channel)

        # flush
        channel.log()

        # all done; indicate success
        return 0


# end of file
