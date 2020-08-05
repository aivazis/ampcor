// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_dom_Offsets_h)
#define ampcor_dom_Offsets_h


// specification of the offset field between a pair of SLC rasters
// it's a map (r_x, r_y) -> (s_x, s_y) of indices of a reference raster to indices of a
// secondary raster

// it is laid out as two blocks of pairs of floats, with the reference indices in the first
// block, and the corresponding secondary indices in the second; the indices are floats because
// the algorithm can achieve sub-pixel resolution

// the indices are laid out as a {m_x, m_y} 2-d grid because in most use cases the reference
// points are generated by some 2-d strategy that attempts to cover the reference image; so
// {offsets_t} is a {2, m_x, m_y} grid
class ampcor::dom::Offsets {
    // types
public:
    // me
    using offsets_type = Offsets;
    // my parts
    using pixel_type = std::pair<float, float>;
    // my layout
    using layout_type = layout_t<3>;
    using layout_const_reference = const layout_type &;
    // size of things
    using size_type = typename layout_type::size_type;

    // metamethods
public:
    // whole raster specification
    constexpr Offsets(layout_const_reference);

    // interface
public:
    // the memory footprint; the cell capacity is given by the inherited {cells}
    constexpr auto cells() const -> size_type;
    constexpr auto bytes() const -> size_type;

    constexpr auto layout() const -> layout_type;
    constexpr auto shape() const -> layout_type::shape_type;

    // implementation details: data
private:
    const layout_type _layout;
};


// get the inline definitions
#define ampcor_dom_Offsets_icc
#include "Offsets.icc"
#undef ampcor_dom_Offsets_icc


#endif

// end of file
