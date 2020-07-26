# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# externals
import journal
import ampcor


# declaration
class Offsets(ampcor.shells.command, family="ampcor.cli.offsets"):
    """
    Estimate an offset field given a pair of rasters
    """


    # user configurable state
    # the input data products
    reference = ampcor.dom.raster()
    reference.doc = "the reference raster image"

    secondary = ampcor.dom.raster()
    secondary.doc = "the secondary raster image"

    # the output data product
    offsets = ampcor.dom.offsets()
    offsets.doc = "the offset map from the reference to the secondary raster"

    # the factory
    correlator = ampcor.correlators.correlator()
    correlator.doc = "the calculator of the offset field"


    # behaviors
    @ampcor.export(tip="produce the offset field between the {reference} and {secondary} rasters")
    def estimate(self, plexus, **kwds):
        """
        Produce an offset map between the {reference} and {secondary} images
        """
        # get the reference image
        reference = self.reference
        # the secondary image
        secondary = self.secondary
        # the output offset map
        offsets = self.offsets
        # and the correlator
        correlator = self.correlator
        # ask the correlator to do its thing
        return correlator.estimate(plexus=plexus,
                                   reference=reference, secondary=secondary, offsets=offsets,
                                   **kwds)


    @ampcor.export(tip="display my configuration")
    def info(self, plexus, **kwds):
        """
        Display the action configuration
        """
        # configure the workflow
        self.flow()

        # grab a channel
        channel = journal.info("ampcor.info")
        # get things going
        channel.line()
        # get the report
        doc = "\n".join(self.show(plexus=plexus))
        # and print it
        channel.log(doc)
        # all done; indicate success
        return 0


    # implementation details
    def show(self, plexus, indent=" "*4, margin=""):
        """
        Generate a configuration report
        """
        # get shell
        shell = plexus.shell
        # unpack my arguments
        reference = self.reference
        secondary = self.secondary
        offsets = self.offsets
        correlator = self.correlator

        # show the shell configuration
        yield f"{margin}shell: {plexus.shell}"
        yield f"{margin}{indent}hosts: {plexus.shell.hosts}"
        yield f"{margin}{indent}tasks: {plexus.shell.tasks} per host"
        yield f"{margin}{indent}gpus:  {plexus.shell.gpus} per task"

        # inputs
        yield f"{margin}input rasters:"

        # reference raster
        yield f"{margin}{indent}reference: {reference}"
        # if i have one
        if reference:
            yield f"{margin}{indent*2}data: {reference.data}"
            yield f"{margin}{indent*2}shape: {reference.shape}"
            yield f"{margin}{indent*2}pixels: {reference.cells()}"
            yield f"{margin}{indent*2}footprint: {reference.bytes()} bytes"

        # secondary raster
        yield f"{margin}{indent}secondary: {secondary}"
        # if i have one
        if secondary:
            yield f"{margin}{indent*2}data: {secondary.data}"
            yield f"{margin}{indent*2}shape: {secondary.shape}"
            yield f"{margin}{indent*2}pixels: {secondary.cells()}"
            yield f"{margin}{indent*2}footprint: {secondary.bytes()} bytes"

        # the output
        yield f"{margin}output:"
        yield f"{margin}{indent}offsets: {offsets}"
        # if i have one
        if offsets:
            # show me
            yield f"{margin}{indent*2}shape: {offsets.shape}"

        # the factory
        yield from correlator.show(indent=indent, margin=margin)

        # all done
        return


    def flow(self):
        """
        Assemble the workflow
        """
        # unpack my products
        reference = self.reference
        secondary = self.secondary
        offsets = self.offsets
        # and my factory
        correlator = self.correlator

        # configure the workflow
        # set up the inputs
        correlator.reference = reference
        correlator.secondary = secondary
        # attach the output
        correlator.offsets = offsets

        # and return the factory
        return correlator


# end of file
