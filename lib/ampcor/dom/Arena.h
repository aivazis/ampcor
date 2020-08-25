// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_dom_Arena_h)
#define ampcor_dom_Arena_h


// an arena is a container of detected SLC tiles
template <typename pixelT>
class ampcor::dom::Arena {
    // types
public:
    // me
    using arena_type = Arena;
    // my pixel type
    using pixel_type = pixelT;
    // my layout
    using layout_type = layout_t<3>;
    using layout_const_reference = const layout_type &;

    // metamethods
public:
    constexpr Arena(layout_const_reference);

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
#define ampcor_dom_Arena_icc
#include "Arena.icc"
#undef ampcor_dom_Arena_icc


#endif

// end of file
