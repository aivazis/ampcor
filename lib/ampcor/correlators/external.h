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
#include <valarray>
// externals
#include <p2/grid.h>

// pull the product definitions
#include <ampcor/dom.h>

// type aliases
namespace ampcor::correlators {
    // strings
    using string_t = std::string;

    // complex numbers
    template <typename T>
    using complex_t = std::complex<T>;
}


#endif

// end of file
