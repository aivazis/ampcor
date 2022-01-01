# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2022 all rights reserved
#


# get the package
import ampcor


# the protocols
from .Domain import Domain as domain
from .Functor import Functor as functor
from .Cover import Cover as cover


# correlation strategies
@ampcor.foundry(implements=ampcor.specs.correlator,
                tip="estimate an offset field using MGA's implementation")
def mga():
    # get the action
    from .MGA import MGA
    # borrow its doctsring
    __doc__ = MGA.__doc__
    # and publish it
    return MGA


# strategies for placing tiles on the reference and secondary rasters
@ampcor.foundry(implements=cover, tip="a grid based generator of a coarse offset map")
def grid():
    # get the action
    from .Grid import Grid
    # borrow its doctsring
    __doc__ = Grid.__doc__
    # and publish it
    return Grid


# strategies for laying tiles on the reference raster
@ampcor.foundry(implements=domain, tip="generate a uniform grid of reference points")
def uniform():
    # get the action
    from .UniformGrid import UniformGrid
    # borrow its doctsring
    __doc__ = UniformGrid.__doc__
    # and publish it
    return UniformGrid


# generators of points on the secondary raster
@ampcor.foundry(implements=functor, tip="a functor that applies a constant shift")
def constant():
    # get the action
    from .Constant import Constant
    # borrow its doctsring
    __doc__ = Constant.__doc__
    # and publish it
    return Constant


# programmatic access to the entities in this package
def newMGA(**kwds):
    """
    Create a new {mga} correlator
    """
    # get the correlator
    from .MGA import MGA
    # instantiate one and return it
    return MGA(**kwds)


def newGrid(**kwds):
    """
    Create a new grid based point generator
    """
    # get the generator
    from .Grid import Grid
    # instantiate one and return it
    return Grid(**kwds)


def newUniformGrid(**kwds):
    """
    A point generator that constructs a uniformly spaced grid
    """
    # get the generator
    from .UniformGrid import UniformGrid
    # instantiate one and return it
    return UniformGrid(**kwds)


def newConstant(**kwds):
    """
    A functor that applies a fixed offset
    """
    # get the functor
    from .Constant import Constant
    # instantiate one and return it
    return Constant(**kwds)


# end of file
