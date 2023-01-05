# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2023 all rights reserved
#


# the package
import ampcor


# ampcor is a workflow
class Ampcor(ampcor.flow.workflow,
              family="ampcor.workflows.ampcor", implements=ampcor.specs.ampcor):
    """
    Produce an offset map from a reference to a secondary raster
    """


    # the input data products
    reference = ampcor.specs.slc()
    reference.doc = "the reference raster image"

    secondary = ampcor.specs.slc()
    secondary.doc = "the secondary raster image"

    # the output data product
    offsetMap = ampcor.specs.offsets()
    offsetMap.doc = "the offset map from the reference to the secondary raster"

    # the factory
    correlator = ampcor.specs.correlator()
    correlator.doc = "the calculator of the offset field"


    # metamethods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # set up the workflow
        self.bind()
        # all done
        return


    # implementation details
    def bind(self):
        """
        Wire up my factory to its inputs and outputs
        """
        # unpack the inputs and outputs
        ref = self.reference
        sec = self.secondary
        map = self.offsetMap

        # get the factory
        cor = self.correlator
        # bind
        cor.reference = ref
        cor.secondary = sec
        cor.offsets = map

        # all done
        return


    def show(self, indent, margin):
        """
        Generate a report of my configuration
        """
        # sign in
        yield f"{margin}name: {self.pyre_name}"
        yield f"{margin}family: {self.pyre_family()}"

        # unpack
        reference = self.reference
        secondary = self.secondary
        offsets = self.offsetMap
        correlator = self.correlator

        # inputs
        yield f"{margin}input rasters:"

        # if i have a reference raster
        if reference:
            # show me
            yield f"{margin}{indent}reference:"
            yield from reference.show(margin=margin+indent*2, indent=indent)

        # if i have a secondary raster
        if secondary:
            # show me
            yield f"{margin}{indent}secondary:"
            yield from secondary.show(margin=margin+indent*2, indent=indent)

        # the output
        yield f"{margin}output:"
        # if i have one
        if offsets:
            # show me
            yield f"{margin}{indent}offsets:"
            yield from offsets.show(margin=margin+indent*2, indent=indent)

        # the factory
        yield from correlator.show(indent=indent, margin=margin)

        # all done
        return


# end of file
