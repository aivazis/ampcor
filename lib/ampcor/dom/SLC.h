// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_dom_SLC_h)
#define ampcor_dom_SLC_h


// specification of an SLC product; here we are concerned about its layout and its pixel type;
// determining the storage type is {Product}'s responsibility
class ampcor::dom::SLC : public layout_t {
    // types
public:
    // my parts
    using pixel_type = complex_t<float>;

    // static interface
public:
    // the memory footprint of my pixels
    static constexpr auto pixelFootprint() -> size_t;
};


// get the inline definitions
#define ampcor_dom_SLC_icc
#include "SLC.icc"
#undef ampcor_dom_SLC_icc


#endif

// end of file
