#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 1998-2025 all rights reserved


# get the package
import ampcor

# make an execution environment
app = ampcor.shells.ampcor(name='plexus')
# configure its layout
app.shell.hosts = 1
app.shell.tasks = 1
app.shell.gpus = 0

# workspace management
ws = ampcor.executive.fileserver
# mount the data directory
ws["alos"] = ampcor.filesystem.local(root="../../../data/alos").discover()
# mount the current working directory
ws["startup"] = ampcor.filesystem.local(root=".").discover()
# realize the output file
ws["startup"].touch(name="quad.dat")
# show me the contents
# print("\n".join(ws.dump()))
# raise SystemExit(0)

# data products
# the reference SLC
ref = ampcor.products.newSLC(name="20061231")
# its shape
ref.shape = 36864,10344
# the path to the data file
ref.data = f"/alos/20061231.slc"

# the secondary SLC
sec = ampcor.products.newSLC(name="20070215")
# its shape
sec.shape = 36864,10344
# the path to the data file
sec.data = f"/alos/20070215.slc"

# the output
offsets = ampcor.products.newOffsets(name="quad")
# the plan shape; determines how many correlation pairs will be computed
offsets.shape = 1,1
# the path to the output file
offsets.data = f"/startup/{offsets.pyre_name}.dat"

# the generator of points on the reference image
# {uniformGrid} is the default, so this is not strictly speaking necessary
# it doesn't have any user configurable state either
uniform = ampcor.correlators.newUniformGrid(name="uni")

# the functor that maps points in the reference image onto the secondary image
# used by the tile pair generator below to place tiles in the secondary image
constant = ampcor.correlators.newConstant(name="zero")
# you can use this to apply a constant shift
constant.shift = 0,0

# these two together become the strategy for pairing up points in the reference image
# to points in the secondary image to form tiles
grid = ampcor.correlators.newGrid(name="")
# the strategy for generating points in the reference image
grid.domain = uniform
# and mapping these onto points in the secondary image
grid.functor = constant

# the correlator
mga = ampcor.correlators.newMGA(name="large")
# the inputs
mga.reference = ref
mga.secondary = sec
# the output
mga.offsets = offsets
# the generator of pairs of points in the reference and secondary images
mga.cover = grid
# configuration
# reference chip size
mga.chip = 128, 128
mga.chip = 16, 16
# padding used to form the search window in the secondary raster
mga.padding = 32, 32
mga.padding = 8, 8
# the refinement factor
mga.refineFactor = 4
# the refinement margin
mga.refineMargin = 8
# the zoom factor
mga.zoomFactor = 8
# show me
# print("\n".join(mga.show(indent=" "*2, margin="")))
# raise SystemExit(0)

# invoke the correlator
mga.estimate(plexus=app)


# end of file
