// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2025 all rights reserved

// code guard
#if !defined(ampcor_dom_Offsets_h)
#define ampcor_dom_Offsets_h


// specification of the offset field between a pair of SLC rasters
// the map is a 2d grid of the {pixel_type} struct:
// - a pair floats with the coordinates in the reference raster
// - a pair floats with the deltas to the best match in the secondary raster
// - a float that reflects the confidence in the mapping and can act as a mask
// - and the snr and covariance of the best match
// for a total of seven fields
class ampcor::dom::Offsets {
    // types
public:
    // me
    using offsets_type = Offsets;
    // my parts
    struct pixel_type;
    // my layout
    using layout_type = layout_t<2>;
    using layout_const_reference = const layout_type &;

    // metamethods
public:
    // whole raster specification
    constexpr Offsets(layout_const_reference);

    // interface
public:
    // my layout
    constexpr auto layout() const -> layout_const_reference;
    // the memory footprint; the cell capacity is given by the inherited {cells}
    constexpr auto cells() const -> std::size_t;
    constexpr auto bytes() const -> std::size_t;

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
