// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2025 all rights reserved

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

    // my layout is the cartesian product of
    using id_layout_type = layout_t<1>;
    // with the layout of an {slc} tile
    using slc_layout_type = SLC::layout_type;

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

    // build the layout of a specific tile
    constexpr auto tile(int tid) const -> layout_type;

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
