// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_viz_SLCPhaser_h)
#define ampcor_viz_SLCPhaser_h


// a microsoft bitmap generator
class ampcor::viz::SLCPhaser {
    // types
public:
    // the slc raster
    using slc_t = ampcor::dom::slc_const_raster_t;
    // iterator over SLC raster pixels
    using source_type = slc_t::const_iterator;
    // and a reference to one
    using source_reference = source_type &;

    // the SLC pixel type
    using pixel_type = slc_t::spec_type::pixel_type;
    // and its support
    using value_type = slc_t::spec_type::value_type;

    // metamethods
public:
    // constructor
    inline SLCPhaser(source_reference source);

    // interface
public:
    inline auto operator*() -> value_type;
    inline auto operator++() -> void;

    // implementation details: data
private:
    source_reference _source;
    const value_type _deg = 180 / (4.*std::atan(1.));

    // default metamethods
public:
    // destructor
    ~SLCPhaser() = default;

    // deleted metamethods
private:
    // constructors
    SLCPhaser(const SLCPhaser &) = delete;
    SLCPhaser & operator=(const SLCPhaser &) = delete;
    inline SLCPhaser(SLCPhaser &&) = delete;
    SLCPhaser & operator=(SLCPhaser &&) = delete;
};


// get the inline definitions
#define ampcor_viz_SLCPhaser_icc
#include "SLCPhaser.icc"
#undef ampcor_viz_SLCPhaser_icc


#endif

// end of file
