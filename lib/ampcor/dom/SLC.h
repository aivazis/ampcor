// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_dom_SLC_h)
#define ampcor_dom_SLC_h


// specification of an SLC product
class ampcor::dom::SLC {
    // types
public:
    // me
    using slc_type = SLC;
    // my pixels are complex
    using value_type = float;
    using pixel_type = complex_t<value_type>;
    // my layout
    using layout_type = layout_t<2>;
    using layout_const_reference = const layout_type &;

    // metamethods
public:
    constexpr SLC(layout_const_reference);

    // interface
public:
    // my layout
    constexpr auto layout() const -> layout_const_reference;
    // my footprint
    constexpr auto cells() const -> std::size_t;
    constexpr auto bytes() const -> std::size_t;

    // implementation details: data
private:
    const layout_type _layout;
};


// get the inline definitions
#define ampcor_dom_SLC_icc
#include "SLC.icc"
#undef ampcor_dom_SLC_icc


#endif

// end of file
