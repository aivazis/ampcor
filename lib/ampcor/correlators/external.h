// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_correlators_external_h)
#define ampcor_correlators_external_h


// STL
#include <complex>
#include <algorithm>
// externals
#include <p2/grid.h>


// type aliases
namespace ampcor::correlators {
    // the sizes of things
    using size_t = std::size_t;

    // complex numbers
    template <typename T>
    using complex_t = std::complex<T>;
}


#endif

// end of file
