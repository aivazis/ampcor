# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#


# pull the action protocol
from ..shells import action
# and the base panel
from ..shells import command
# pull in the command decorator
from .. import foundry


# commands
@foundry(implements=action, tip="estimate an offset field given a pair of raster images")
def offsets():
    # get the action
    from .Offsets import Offsets
    # borrow its doctsring
    __doc__ = Offsets.__doc__
    # and publish it
    return Offsets


# help
@foundry(implements=action, tip="display information about this application")
def about():
    # get the action
    from .About import About
    # borrow its docstring
    __doc__ = About.__doc__
    # and publish it
    return About


@foundry(implements=action, tip="display configuration information about this application")
def config():
    # get the action
    from .Config import Config
    # borrow its docstring
    __doc__ = Config.__doc__
    # and publish it
    return Config


@foundry(implements=action, tip="display debugging information about this application")
def debug():
    # get the action
    from .Debug import Debug
    # borrow its docstring
    __doc__ = Debug.__doc__
    # and publish it
    return Debug


# command completion; no tip so it doesn't show up on the help panel
@foundry(implements=action)
def complete():
    # get the action
    from .Complete import Complete
    # and publish it
    return Complete


# end of file
