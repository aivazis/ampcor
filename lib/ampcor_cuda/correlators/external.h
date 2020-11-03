// -*- C++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// code guard
#if !defined(ampcor_cuda_correlators_external_h)
#define ampcor_cuda_correlators_external_h


// STL
#include <cmath>
#include <complex>
#include <algorithm>
#include <functional>
#include <numeric>
#include <exception>
// cuda
#include <cuda_runtime.h>
#include <cuComplex.h>
// pyre
#include <pyre/journal.h>
#include <p2/timers.h>
#include <p2/grid.h>

// access to the dom
#include <ampcor/dom.h>

// type aliases
namespace ampcor::cuda::correlators {
    // local type aliases
    // sizes of things
    using size_t = std::size_t;

    // pyre timers
    using timer_t = pyre::timers::process_timer_t;
}


#endif

// end of file
