# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2022 all rights reserved
#


# externals
import cuda
import journal
import math
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
        Compute the portion of the offset map between a pair of rasters given by {box}
        """
        # get the {grid} bindings
        libgrid = ampcor.libpyre.grid
        # so we can grab the constructors for {layout}
        packing2d = libgrid.Canonical2D
        # {index}
        index2d = libgrid.Index2D
        # and {shape}
        shape2d = libgrid.Shape2D

        # make a timer
        timer = ampcor.executive.newTimer(name="ampcor.cuda.sequential")
        # and a journal channel
        channel = journal.info("ampcor.cuda.timings.sequential")

        # grab a device
        device = cuda.manager.devices[-1]
        # make it the active one
        cuda.manager.device(did=device.id)

        # compute the memory requirements
        required = 4*self.plan.arena(box=box)
        # figure out how much memory we have
        available = 0.9 * device.globalMemory
        # compute the number of batches, assuming memory is the limiting resource
        batches = math.ceil(required/available)

        # get the starting row of the plan
        rowOrigin = box.origin[0]
        # get the shape of the rows
        rowShape = box.shape[0]
        # get the row index of one passed the last one
        rowEnd = rowOrigin + rowShape
        # compute the row step
        rowStep = rowShape // batches

        # go through the {box} in batches
        for row in range(rowOrigin, rowEnd, rowStep):
            # form an index that points to the beginning of this batch
            origin = index2d(index=(row, 0))
            # and a shape that covers this batch
            shape = shape2d(shape=(min(rowStep, rowEnd-row), box.shape[1]))
            # use them to specify the workload
            workload = packing2d(origin=origin, shape=shape)
            # do the work
            self.worker.adjust(box=workload)

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

        # save my rank
        self.rank = rank
        # and the plan
        self.plan = plan

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
        seq = ampcor.ext.libampcor_cuda.Sequential
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
