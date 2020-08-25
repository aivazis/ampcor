// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_dom_forward_h)
#define ampcor_dom_forward_h


// set up the namespace
namespace ampcor::dom {
    // the layout specification for the various product types
    // inputs and outputs
    class SLC;
    class Offsets;
    // intermediate
    template <typename pixelT>
    class Arena;

    // turning specs into actual products
    template <class specT, bool isReadOnly>
    class Product;
}


#endif

// end of file
