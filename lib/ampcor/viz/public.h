// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2025 all rights reserved

// code guard
#if !defined(ampcor_viz_public_h)
#define ampcor_viz_public_h


// external packages
#include "external.h"
// set up the namespace
#include "forward.h"

// published type aliases and function declarations that constitute the public API of this package
// this is the file you are looking for
#include "api.h"

// local entities
#include "kernels.h"
// the microsoft BMP generator
#include "BMP.h"
// interpolators that map 1D data to color values
#include "Phase1D.h"
#include "Uniform1D.h"
#include "Complex.h"

// a detector for SLC pixels
#include "SLCDetector.h"
// phase calculator
#include "SLCPhaser.h"

# endif

// end of file
