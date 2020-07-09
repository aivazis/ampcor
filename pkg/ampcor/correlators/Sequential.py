# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


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
    def adjust(self, rasters, plan, **kwds):
        """
        Compute the offset map between a pair of {rasters} given a correlation {plan}
        """
        # unpack the raster
        ref, sec = rasters
        # ask the plan for the number of points in the domain
        points = len(plan)
        # get the shape of the reference chip
        chip = plan.chip
        # and the shape of the search windows
        window = plan.window

        # access the bindings; this is guaranteed to succeed
        libampcor = ampcor.ext.libampcor
        # instantiate my worker
        worker = libampcor.Sequential(points, chip, window)

        # all done
        return



# end of file
