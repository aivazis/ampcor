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


# declaration
class MGA(ampcor.flow.factory,
          family="ampcor.correlators.mga", implements=ampcor.specs.correlator):
    """
    MGA's implementation of the offset field estimator
    """


    # user configurable state
    # control over the placement of the initial guesses
    cover = ampcor.correlators.cover()
    cover.doc = "the strategy for generating the initial guesses"

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

    # inputs
    reference = ampcor.specs.slc.input()
    reference.doc = "the reference raster"

    secondary = ampcor.specs.slc.input()
    secondary.doc = "the secondary raster"

    # output
    offsets = ampcor.specs.offsets.output()
    offsets.doc = "the offset map from the reference to the secondary raster"


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
        # get the coarse map
        map = self.cover.map(bounds=reference.shape, shape=offsets.shape)
        # initialize the output product
        self.primeOffsets(map=map)
        # stop the timer
        timer.stop()
        # show me
        channel.log(f"primed the offset map: {1e3 * timer.read():.3f} ms")

        # start the timer
        timer.reset().start()
        # make a plan
        plan = self.newPlan(correlator=self, regmap=map, rasters=(reference,secondary))
        # stop the timer
        timer.stop()
        # show me
        channel.log(f"correlation plan: {1e3 * timer.read():.3f} ms")

        # restart the timer
        timer.reset().start()
        # open the two rasters and get access to the data
        ref = self.reference.open()
        sec = self.secondary.open()
        # stop the timer
        timer.stop()
        # show me
        channel.log(f"opened the two rasters: {1e3 * timer.read():.3f} ms")

        # restart the timer
        timer.reset().start()
        # choose the correlator implementation
        worker = self.makeWorker(rasters=(ref, sec), offsets=offsets,
                                 layout=plexus.shell, plan=plan)
        # compute the offsets
        worker.adjust(box=offsets.layout)
        # stop the timer
        timer.stop()
        # show me
        channel.log(f"offset field: {1e3 * timer.read():.3f} ms")

        # all done
        return 0


    # interface
    def primeOffsets(self, map):
        """
        Initialize the offset map by recording the initial guesses
        """
        # get the output product and create its raster
        offsets = self.offsets.open(mode="n")

        # go through the pairs of corresponding points in tandem
        for pair, (p, q) in enumerate(zip(*map)):
            # form the shift that takes {p} into {q}
            delta = q[0]-p[0], q[1]-p[1]
            # get the entry
            rec = offsets[pair]
            # store the original point
            rec.ref = p
            # store the shift
            rec.delta = delta
            # zero out the other fields
            rec.confidence = 0
            rec.snr = 0
            rec.covariance = 0

        # all done
        return


    def makeWorker(self, rasters, offsets, layout, plan):
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
        worker = workerFactory(rasters=rasters, offsets=offsets, manager=self, plan=plan)
        # that's all until there is support for other types of parallelism
        return worker


    def show(self, indent, margin):
        """
        Generate a report of my configuration
        """
        # show who i am
        yield f"{margin}estimator:"
        yield f"{margin}{indent}name: {self.pyre_name}"
        yield f"{margin}{indent}family: {self.pyre_family()}"
        # display the reference chip size
        yield f"{margin}{indent}chip: {self.chip}"
        # the search window padding
        yield f"{margin}{indent}padding: {self.padding}"
        # the refinement factor
        yield f"{margin}{indent}refinement factor: {self.refineFactor}"
        # the refinement margin
        yield f"{margin}{indent}refinement margin: {self.refineMargin}"
        # the zoom factor
        yield f"{margin}{indent}zoom factor: {self.zoomFactor}"

        # describe my coarse map strategy
        yield from self.cover.show(indent, margin=margin+indent)

        # unpack the rasters
        reference = self.reference
        secondary = self.secondary
        # grab the output
        offsets = self.offsets
        # get the coarse map
        map = self.cover.map(bounds=reference.shape, shape=offsets.shape)
        # make a plan
        plan = self.newPlan(correlator=self, regmap=map, rasters=(reference,secondary))
        # and show me the plan details
        yield from plan.show(indent=indent, margin=margin+indent)

        # all done
        return


    # flow hooks
    @ampcor.export
    def pyre_make(self, **kwds):
        """
        Build my outputs
        """
        # invoke me
        # NYI: automate this
        return self.estimate(**kwds)


    # private data
    timer = ampcor.executive.newTimer(name="ampcor.mga")


# end of file
