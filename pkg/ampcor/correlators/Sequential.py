# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2021 all rights reserved
#


# externals
import journal
# get the package
import ampcor


# worker that goes through the tiles in the execution plan sequentially
class Sequential:
    """
    The sequential tile registration strategy
    """


    # interface
    def adjust(self, box, **kwds):
        """
        Compute the offset map between a pair of rasters given a correlation {plan}
        """
        # make a timer
        timer = ampcor.executive.newTimer(name="ampcor.sequential")
        # and a journal channel
        channel = journal.info("ampcor.timings.sequential")

        # start the timer
        timer.reset().start()
        # compute the adjustments to the offset field
        self.worker.adjust(box=box)
        # stop the timer
        timer.stop()
        # show me
        channel.log(f"[{self.rank}]: computed offsets: {1e3 * timer.read():.3f} ms")

        # all done
        return


    # metamethods
    def __init__(self, rasters, offsets, manager, plan, rank=0, **kwds):
        # chain up
        super().__init__(**kwds)

        # save my rank
        self.rank = rank

        # unpack the rasters
        ref, sec = rasters
        # get the shape of the reference chip
        chip = plan.chip
        # and the shape of the search windows
        window = plan.window

        # the manager holds the refinement plan
        refineFactor = manager.refineFactor
        refineMargin = manager.refineMargin
        zoomFactor = manager.zoomFactor

        # make a timer
        timer = ampcor.executive.newTimer(name="ampcor.sequential")
        # and a journal channel
        channel = journal.info("ampcor.timings.sequential")

        # access the bindings; this is guaranteed to succeed
        seq = ampcor.ext.libampcor.Sequential
        # start the timer
        timer.reset().start()

        # instantiate my worker
        self.worker = seq(rank=rank,
                          reference=ref.raster, secondary=sec.raster, map=offsets.raster,
                          chip=chip, window=window,
                          refineFactor=refineFactor, refineMargin=refineMargin,
                          zoomFactor=zoomFactor)
        # stop the timer
        timer.stop()
        # show me
        channel.log(f"[{rank}]: instantiated a sequential worker: {1e3 * timer.read():.3f} ms")

        # all done
        return


# end of file
