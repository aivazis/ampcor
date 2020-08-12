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
        Correlate a pair of {rasters} given a collection of {tiles}
        """
        # unpack the rasters
        ref, sec = rasters
        # realize the {tiles}, just in case what came in was a generator
        tiles = plan.tiles
        # because we need to know how many pairs there are
        pairs = len(tiles)
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
        worker = libampcor.Sequential(pairs=pairs,
                                      ref=chip, sec=window,
                                      refineFactor=refineFactor, refineMargin=refineMargin,
                                      zoomFactor=zoomFactor
                                      )

        # show me
        channel.log("loading tiles")
        # go through the valid pairs of reference and secondary tiles
        for idx, (pid, r, s) in enumerate(tiles):
            # save the pair id
            worker.addPair(tid=idx, pid=pid)
            # load the reference tile
            worker.addReferenceTile(tid=idx, raster=ref, tile=r)
            # load the secondary tile
            worker.addSecondaryTile(tid=idx, raster=sec, tile=s)

        channel.log("adjusting the offset map")
        # ask the worker to perform pixel level adjustments
        worker.adjust(offsets)

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
        # manager.device(did=4)
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
