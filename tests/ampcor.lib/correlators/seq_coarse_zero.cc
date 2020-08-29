// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// support
#include <cassert>
// get the headers
#include <ampcor/dom.h>
#include <ampcor/correlators.h>
// the plan details
#include "plan.h"


// type aliases
// products
using slc_const_raster_t = ampcor::dom::slc_const_raster_t;
using offsets_raster_t = ampcor::dom::offsets_raster_t;
// the correlator
using seq_t = ampcor::correlators::sequential_t<slc_const_raster_t, offsets_raster_t>;


// make a sequential worker and add the reference tiles to its arena
int main(int argc, char *argv[]) {
    // initialize the journal
    pyre::journal::init(argc, argv);
    pyre::journal::application("seq_coarse_zero");
    // make a channel
    pyre::journal::debug_t channel("ampcor.correlators.seq");

    // the number of pairs
    auto pairs = plan.pairs;
    // the base tile dimension
    auto dim = plan.dim;

    // the shape of the reference tile arena
    seq_t::arena_shape_type refShape { pairs, dim/4, dim/4 };
    // their layout
    seq_t::arena_layout_type refLayout { refShape };

    // the shape of the secondary tile arena
    seq_t::arena_shape_type secShape { pairs, dim/2, dim/2 };
    // the origin: chosen such that the reference tile shape is at {0,0}
    seq_t::arena_index_type secOrigin { 0, -dim/8, -dim/8 };
    // their layout
    seq_t::arena_layout_type secLayout { secShape, secOrigin };

    // specify and open the input rasters
    // shape
    slc_const_raster_t::shape_type rasterShape { dim, dim };
    // product spec
    slc_const_raster_t::spec_type spec { slc_const_raster_t::layout_type(rasterShape) };
    // open the sample reference raster in read-only mode
    slc_const_raster_t ref { spec, "slc_ref.dat" };
    // repeat for the secondary raster
    slc_const_raster_t sec { spec, "slc_sec.dat" };

    // the output is a 2x2 grid
    offsets_raster_t::shape_type offsetShape { 2, 2 };
    // build the spec of the output product
    offsets_raster_t::spec_type offsetSpec { offsets_raster_t::layout_type(offsetShape) };
    // open the output
    offsets_raster_t offsets { offsetSpec, "offsets.dat", offsetSpec.cells() };

    // make a sequential worker
    seq_t seq(ref, sec, offsets, refLayout, secLayout, 1, 0, 1);

    // all we know is the number of pairs
    assert(( seq.pairs() == pairs ));

    // all done
    return 0;
}


// end of file
