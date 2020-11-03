# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# externals
import journal
# framework
import ampcor


# declaration
class CUDA:
    """
    The CUDA accelerated registration strategy
    """


    # interface
    def adjust(self, box, **kwds):
        """
        Compute the offset map between a pair of rasters given a correlation {plan}
        """
        # make a timer
        timer = ampcor.executive.newTimer(name="ampcor.cuda.sequential")
        # and a journal channel
        channel = journal.info("ampcor.cuda.timings.sequential")

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


    # meta-methods
    def __init__(self, rasters, offsets, manager, plan, rank=0, **kwds):
        # chain up
        super().__init__(**kwds)

        # get the cuda support
        import cuda
        # grab a device
        cuda.manager.device(did=0)

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
        timer = ampcor.executive.newTimer(name="ampcor.cuda.sequential")
        # and a journal channel
        channel = journal.info("ampcor.cuda.timings.sequential")

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
        channel.log(f"[{rank}]: instantiated a cuda sequential worker: {1e3 * timer.read():.3f} ms")

        # all done
        return


    def __del__(self):
        # get the cuda support
        import cuda
        # get the cuda device manager
        manager = cuda.manager
        # and reset the device
        manager.reset()
        # all done
        return


# end of file
