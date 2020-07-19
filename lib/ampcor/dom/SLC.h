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
class ampcor::dom::SLC : public layout_t<2> {
    // types
public:
    // me
    using slc_type = SLC;
    // my base
    using layout_type = layout_t<2>;
    using layout_const_reference = const layout_type &;
    // my parts
    using value_type = float;
    using pixel_type = complex_t<value_type>;
    // my shape
    using shape_type = typename layout_type::shape_type;
    using shape_const_reference = const shape_type &;
    // my indices
    using index_type = typename layout_type::index_type;
    using index_const_reference = const index_type &;
    // size of things
    using size_type = typename shape_type::size_type;

    // metamethods
public:
    // whole raster specification
    constexpr SLC(shape_const_reference);
    // for tiles
    constexpr SLC(layout_const_reference);

    // interface
public:
    constexpr auto bytes() const -> size_type;

    // slice factory
public:
    constexpr auto tile(index_const_reference, shape_const_reference) const -> slc_type;

    // static interface
public:
    // the memory footprint of my pixels
    static constexpr auto bytesPerCell() -> size_type;
};


// get the inline definitions
#define ampcor_dom_SLC_icc
#include "SLC.icc"
#undef ampcor_dom_SLC_icc


#endif

// end of file
