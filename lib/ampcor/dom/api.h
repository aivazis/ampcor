// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_dom_api_h)
#define ampcor_dom_api_h


// publicly visible types
namespace ampcor::dom {
    // slc
    // the spec
    using slc_t = SLC;
    // the rasters
    using slc_raster_t = Product<SLC, false>;              // read/write
    using slc_const_raster_t = Product<SLC, true>;         // read only

    // offset maps
    // the spec
    using offsets_t = Offsets;
    // the rasters
    using offsets_raster_t = Product<Offsets, false>;      // read/write
    using offsets_const_raster_t = Product<Offsets, true>; // read only
}

#endif

// end of file
