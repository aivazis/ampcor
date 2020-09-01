// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// plan details
struct {
    // the shape of the non-trivial part of the reference tiles
    const ampcor::dom::slc_raster_t::shape_type seedShape { 3, 3 };
    // the margin around the non-trivial part
    const ampcor::dom::slc_raster_t::shape_type seedMargin { 1, 1 };

    // the shape of the offset map
    const ampcor::dom::offsets_raster_t::shape_type gridShape { 2, 2 };

} plan;


// end of file
