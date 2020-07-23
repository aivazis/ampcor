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
// access the complex literals
using namespace std::complex_literals;


// type aliases
using slc_t = ampcor::dom::slc_t;
using seq_t = ampcor::correlators::sequential_t<slc_t>;


// make a sequential worker and add the reference tiles to its arena
int main(int argc, char *argv[]) {
    // initialize the journal
    pyre::journal::init(argc, argv);
    pyre::journal::application("seq_coarse_zero");
    // make a channel
    pyre::journal::debug_t channel("ampcor.correlators.seq");

    // the base dimension
    size_t dim = 8;

    // the shape of the reference raster
    slc_t::shape_type rasterShape { dim, dim };
    // the product spec
    slc_t::spec_type spec { rasterShape };
    // open the sample reference raster in read-only mode
    slc_t slc { spec, "slc_ref.dat" };

    // the shape of the reference tiles
    slc_t::shape_type refShape { dim/4, dim/4 };
    // and their layout: 0-based, row major
    slc_t::layout_type refLayout { refShape };

    // the shape of the secondary tiles, which includes their margin
    slc_t::shape_type secShape { dim/2, dim/2 };
    // the origin: chosen such that the reference tile shape is at {0,0}
    slc_t::index_type secOrigin { -dim/8, -dim/8 };
    // their layout
    slc_t::layout_type secLayout { secShape, secOrigin };

    // make a sequential worker
    seq_t seq(4, refLayout, secLayout, 1, 0, 1);

    // pick a value
    slc_t::spec_type::value_type v = 42;
    // zero out its coarse arena
    seq.fillCoarseArena(v);

    // go through every cell
    for (auto cursor = seq.coarseArena();
         cursor != seq.coarseArena()+seq.coarseArenaCells(); ++cursor) {
        // make sure it contains what we expect
        assert(( *cursor == v ));
    }

    // all done
    return 0;
}


// end of file
