# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2021 all rights reserved
#


# externals
import ampcor
import itertools
import journal


# declaration
class Offsets(ampcor.shells.command, family="ampcor.cli.offsets"):
    """
    Estimate an offset field given a pair of rasters
    """


    # my workflow
    flow = ampcor.specs.ampcor()
    flow.doc = "the ampcor workflow"


    # behaviors
    @ampcor.export(tip="produce the offset field between the {reference} and {secondary} rasters")
    def estimate(self, plexus, **kwds):
        """
        Produce an offset map between the {reference} and {secondary} images
        """
        # invoke the workflow and return the execution status
        return self.flow.pyre_make(plexus=plexus)


    @ampcor.export(tip="display my configuration")
    def info(self, plexus, **kwds):
        """
        Display the action configuration
        """
        # grab a channel
        channel = journal.info("ampcor.info")
        # get things going
        channel.line()
        # get the report
        doc = "\n".join(self.show(plexus=plexus))
        # and print it
        channel.log(doc)
        # all done; indicate success
        return 0


    @ampcor.export(tip="display the contents of the offsets product")
    def dump(self, plexus, **kwds):
        """
        Display the contents of the offsets product
        """
        # make a channel
        channel = journal.info("ampcor.offsets.dump")

        # get the output
        offsets = self.flow.offsetMap
        # open its raster
        offsets.open(mode="r")

        # build a set of indices that visit the map in layout order
        for idx in itertools.product(*map(range, offsets.shape)):
            # get the cell
            p = offsets[idx]
            # show me
            channel.line(f"offsets{idx}:")
            channel.line(f"          ref: {tuple(p.ref)}")
            channel.line(f"        shift: {tuple(p.delta)}")
            channel.line(f"        gamma: {p.gamma}")
            channel.line(f"   confidence: {p.confidence}")
            channel.line(f"          snr: {p.snr}")
            channel.line(f"   covariance: {p.covariance}")
            channel.log()

        # all done
        return 0


    # implementation details
    def show(self, plexus, indent=" "*4, margin=""):
        """
        Generate a configuration report
        """
        # get shell
        shell = plexus.shell
        # get my flow
        flow = self.flow

        # show the shell configuration
        yield f"{margin}shell: {plexus.shell}"
        yield f"{margin}{indent}hosts: {plexus.shell.hosts}"
        yield f"{margin}{indent}tasks: {plexus.shell.tasks} per host"
        yield f"{margin}{indent}gpus:  {plexus.shell.gpus} per task"

        # flow
        yield f"{margin}flow:"
        yield from flow.show(margin=margin+indent, indent=indent)

        # all done
        return


# end of file
