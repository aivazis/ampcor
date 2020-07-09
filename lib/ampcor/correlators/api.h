// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_correlators_api_h)
#define ampcor_correlators_api_h


// publicly visible types
namespace ampcor::correlators {
    // the sequential worker
    template <class productT>
    using sequential_t = Sequential<productT>;
}


#endif

// end of file
