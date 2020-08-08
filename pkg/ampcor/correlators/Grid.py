# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# externals
import itertools
# framework
import ampcor


# declaration
class Grid(ampcor.component,
           family="ampcor.correlators.covers.grid", implements=ampcor.correlators.cover):
    """
    A generator of a Cartesian grid of initial guesses for the offset map
    """


    # user configurable state
    domain = ampcor.correlators.domain()
    domain.doc = "the domain of the map"

    functor = ampcor.correlators.functor()
    functor.doc = "the function that maps points from the reference raster to the secondary raster"


    # requirements
    @ampcor.export
    def map(self, reference, **kwds):
        """
        Build an offset map between {reference} and {secondary}
        """
        # get my domain
        domain = self.domain
        # and the functor that generates the codomain
        functor = self.functor
        # make a map
        offmap = ampcor.dom.newOffsetMap(shape=domain.shape)
        # generate the reference points and attach them as the domain of the offset map
        offmap.domain = tuple(domain.points(bounds=reference.shape))
        # invoke the map to generate the corresponding points on the secondary image
        offmap.codomain = tuple(functor.codomain(domain=offmap.domain))
        # all done
        return offmap


    # interface
    def show(self, indent, margin):
        """
        Display my configuration
        """
        # show who i am
        yield f"{margin}offsets: {self.pyre_family()}"
        # show my domain
        yield from self.domain.show(indent=indent, margin=margin+indent)
        # and my codomain generator
        yield from self.functor.show(indent=indent, margin=margin+indent)
        # all done
        return


# end of file
