# -*- Python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# framework
import ampcor


# declaration
class CUDA:
    """
    The CUDA accelerated registration strategy
    """


    # interface
    def adjust(self, manager, rasters, plan, channel):
        """
        Correlate a pair rasters given a plan
        """
        # unpack the rasters
        ref, sec = rasters
        # ask the plan for the total number of points on the map
        points = len(plan)
        # the shape of the reference chips
        chip = plan.chip
        # and the shape of the search windows
        window = plan.window

        # unpack the refinement margin
        refineMargin = manager.refineMargin
        # the refinement factor
        refineFactor = manager.refineFactor
        # and the zoom factor
        zoomFactor = manager.zoomFactor

        # get the bindings
        libampcor = ampcor.ext.libampcor_cuda
        # make a worker
        worker = libampcor.sequential(points, chip, window, refineMargin, refineFactor, zoomFactor)

        # show me
        channel.log("loading tiles")
        # go through the valid pairs of reference and secondary tiles
        for idx, (r, t) in enumerate(plan.pairs):
            # load the reference slice
            libampcor.addReference(worker, ref, idx, r.begin, r.end)
            # load the secondary slice
            libampcor.addSecondary(worker, sec, idx, t.begin, t.end)

        channel.log("adjusting the offset map")
        # ask the worker to perform pixel level adjustments
        offsets = libampcor.adjust(worker)

        # all done
        return offsets


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # get the cuda support
        import cuda
        # get the cuda device manager
        manager = cuda.manager
        # grab a device
        manager.device(did=4)
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
