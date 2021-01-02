// -*- C++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2021 all rights reserved


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
#include <cufft.h>
#include <cooperative_groups.h>
#include <thrust/complex.h>
// pyre
#include <pyre/journal.h>
#include <pyre/grid.h>
#include <p2/timers.h>

// access to the dom
#include <ampcor/dom.h>

// type aliases
namespace ampcor::cuda::correlators {
    // local type aliases
    // strings
    using string_t = std::string;
    // sizes of things
    using size_t = std::size_t;

    // pyre timers
    using timer_t = pyre::timers::process_timer_t;
}


#endif

// end of file
