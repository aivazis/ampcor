// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_dom_api_h)
#define ampcor_dom_api_h


// publicly visible types
namespace ampcor::dom {
    // slc products
    using slc_t = Product<SLC, false>;        // read/write
    using slc_const_t = Product<SLC, true>;   // read only
}

#endif

// end of file
