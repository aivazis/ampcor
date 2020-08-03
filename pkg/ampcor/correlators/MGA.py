# -*- Python -*-
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
# my protocol
from .Correlator import Correlator


# declaration
class MGA(ampcor.component, family="ampcor.correlators.mga", implements=Correlator):
    """
    MGA's implementation of the offset field estimator
    """


    # input data products
    reference = ampcor.specs.slc()
    reference.doc = "the reference raster image"

    secondary = ampcor.specs.slc()
    secondary.doc = "the secondary raster image"

    # the output data product
    offsets = ampcor.specs.offsets()
    offsets.doc = "the offset map from the reference to the secondary raster"


    # user configurable state
    coarse = ampcor.correlators.offsets()
    coarse.doc = "the initial guess for the offset map"

    # control over the correlation plan
    chip = ampcor.properties.tuple(schema=ampcor.properties.int())
    chip.default = 128, 128
    chip.doc = "the shape of the reference chip"

    padding = ampcor.properties.tuple(schema=ampcor.properties.int())
    padding.default = 32, 32
    padding.doc = "padding around the chip shape to form the search window in the secondary raster"

    refineMargin = ampcor.properties.int(default=8)
    refineMargin.doc = "padding around the anti-aliased search window before the fine adjustments"

    refineFactor = ampcor.properties.int(default=2)
    refineFactor.doc = "anti-aliasing factor for the window with the best coarse correlation"

    zoomFactor = ampcor.properties.int(default=8)
    zoomFactor.doc = "refinement factor for the fine correlation hyper-matrix"


    # types
    from .Plan import Plan as newPlan


    # protocol obligations
    @ampcor.provides
    def estimate(self, plexus, **kwds):
        """
        Estimate the offset field between a pair of raster images
        """
        # make a channel
        channel = journal.info("ampcor.timings")
        # grab my timer
        timer = self.timer

        # show me
        channel.log(f"correlator: {self.pyre_family()}")

        # unpack my products
        # inputs
        reference = self.reference
        secondary = self.secondary
        # outputs
        offsets = self.offsets

        # start the timer
        timer.reset().start()
        # make a plan
        plan = self.makePlan()
        # stop the timer
        timer.stop()
        # show me
        channel.log(f"  correlation plan: {1e3 * timer.read():.3f} ms")

        # restart the timer
        timer.reset().start()
        # open the two rasters and get access to the data
        ref = self.reference.open().raster
        sec = self.secondary.open().raster
        # stop the timer
        timer.stop()
        # show me
        channel.log(f"  opened the two rasters: {1e3 * timer.read():.3f} ms")

        # restart the timer
        timer.reset().start()
        # choose the correlator implementation
        worker = self.makeWorker(layout=plexus.shell)
        # compute the offsets
        regmap = worker.adjust(manager=self, rasters=(ref, sec), offsets=offsets, plan=plan)
        # stop the timer
        timer.stop()
        # show me
        channel.log(f"offset field: {1e3 * timer.read():.3f} ms")

        # all done
        return 0


    # interface
    def makeWorker(self, layout):
        """
        Deduce the correlator implementation strategy
        """
        # if the user asked for GPU acceleration and we support it
        if layout.gpus and ampcor.ext.libampcor_cuda:
            # use the GPU aware implementation
            from .CUDA import CUDA as workerFactory
        # if the CPU implementation is available
        elif ampcor.ext.libampcor:
            # use the native implementation
            from .Sequential import Sequential as workerFactory
        # otherwise
        else:
            # complain
            raise NotImplementedError("no available correlation strategy")

        # instantiate
        worker = workerFactory()
        # that's all until there is support for other types of parallelism
        return worker


    def makePlan(self):
        """
        Formulate a computational plan for correlating {reference} and {secondary} to produce an
        offset map
        """
        # get the inputs
        reference = self.reference
        secondary = self.secondary
        # grab the output
        offsets = self.offsets

        # pair up the rasters
        rasters = reference, secondary
        # get the coarse map
        coarse = self.coarse.map(reference=reference)

        # mark
        raise NotImplementedError("must use the {offsets} product")

        # make a plan
        plan = self.newPlan(correlator=self, regmap=coarse, rasters=rasters)
        # and return it
        return plan


    def show(self, indent, margin):
        """
        Generate a report of my configuration
        """
        # show who i am
        yield f"{margin}estimator: {self.pyre_family()}"
        # display the reference chip size
        yield f"{margin}{indent}chip: {self.chip}"
        # and the search window padding
        yield f"{margin}{indent}padding: {self.padding}"

        # describe my coarse map strategy
        yield from self.coarse.show(indent, margin=margin+indent)

        # MGA: mark
        return

        # make a plan
        plan = self.makePlan()
        # and show me the plan details
        yield from plan.show(indent=indent, margin=margin+indent)

        # all done
        return self


    # private data
    timer = ampcor.executive.newTimer(name="ampcor.mga")


# end of file
