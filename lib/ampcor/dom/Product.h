// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_dom_Product_h)
#define ampcor_dom_Product_h


// this class takes a product layout and builds a memory mapped grid
template <class specT, bool isReadOnly>
class ampcor::dom::Product : public grid_t<specT, mmap_t, isReadOnly> {
    // type aliases
public:
    // my parameters
    using product_type = specT;

    // static interface
public:
    // my read/write flag
    static constexpr auto readOnly() -> bool;
    // my pixel size
    static constexpr auto pixelFootprint() -> size_t;
};


// get the inline definitions
#define ampcor_dom_Product_icc
#include "Product.icc"
#undef ampcor_dom_Product_icc


#endif

// end of file
