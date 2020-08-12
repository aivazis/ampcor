# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# externals
import journal
# get the package
import ampcor


# this worker takes a plan, attempts to allocate enough memory to execute it, and goes through
# each pair of tiles in sequence until they are all done. it is smart about memory allocation,
# in the sense that it will execute the specified plan in batches whose size is determined by
# the available memory
class Sequential:
    """
    The sequential tile registration strategy
    """


    # interface
    def adjust(self, manager, rasters, offsets, plan, **kwds):
        """
        Compute the offset map between a pair of {rasters} given a correlation {plan}
        """
        # make a timer
        timer = ampcor.executive.newTimer(name="ampcor.sequential")
        # and a journal channel
        channel = journal.info("ampcor.timings.sequential")

        # unpack the rasters
        ref, sec = rasters
        # get the tile pairings
        tiles = plan.tiles
        # because we need to know how many there are
        pairs = len(tiles)
        # get the shape of the reference chip
        chip = plan.chip
        # and the shape of the search windows
        window = plan.window

        # the manager holds the refinement plan
        refineFactor = manager.refineFactor
        refineMargin = manager.refineMargin
        zoomFactor = manager.zoomFactor

        # access the bindings; this is guaranteed to succeed
        libampcor = ampcor.ext.libampcor
        # start the timer
        timer.reset().start()
        # instantiate my worker
        worker = libampcor.Sequential(pairs=pairs,
                                      ref=chip, sec=window,
                                      refineFactor=refineFactor, refineMargin=refineMargin,
                                      zoomFactor=zoomFactor
                                      )
        # stop the timer
        timer.stop()
        # show me
        channel.log(f"  instantiated the sequential worker: {1e3 * timer.read():.3f} ms")

        # start the timer
        timer.reset().start()
        # go through the tile pairs
        for idx, (pid, r,s) in enumerate(tiles):
            # save the pair id
            worker.addPair(tid=idx, pid=pid)
            # load the reference tile
            worker.addReferenceTile(tid=idx, raster=ref, tile=r)
            # load the secondary tile
            worker.addSecondaryTile(tid=idx, raster=sec, tile=s)
        # stop the timer
        timer.stop()
        # show me
        channel.log(f"  transferred the tiles to the coarse arena: {1e3 * timer.read():.3f} ms")

        # start the timer
        timer.reset().start()
        # compute the adjustments to the offset field
        # worker.adjust(offsets)
        # stop the timer
        timer.stop()
        # show me
        channel.log(f"  computed the offsets: {1e3 * timer.read():.3f} ms")

        # all done
        return


# end of file
