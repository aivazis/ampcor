# -*- Makefile -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#

# ampcor has a python package
ampcor.packages := ampcor.pkg
# and a ux bundle
ampcor.webpack := ampcor.ux

# other assets
# {libampcor} and its bindings
ampcor.assets := ampcor.lib
# optional cuda acceleration
ampcor.assets += ${if ${filter cuda,$(extern.available)},ampcor_cuda.lib}


# metadata for the ampcor python package
# its name
ampcor.pkg.stem := ampcor
# its plexus wrapper in the {bin} directory
ampcor.pkg.drivers := ampcor


# get the asset definitions
include $(ampcor.assets)

# get the test suites
include $(ampcor.tests)

# end of file
