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
using slc_const_raster_t = ampcor::dom::slc_const_raster_t;
using offsets_raster_t = ampcor::dom::offsets_raster_t;
using seq_t = ampcor::correlators::sequential_t<slc_const_raster_t, offsets_raster_t>;


// make a sequential worker and add tiles to its arena
// - the tiles come from the output of the {slc_ref} and {slc_sec} test cases
// - the reference tiles are (dim/4, dim/4) at the center of (dim/2,dim/2) blocks
// - the secondary tiles are (dim/2,dim/2) and contain the reference tiles at their bottom
//   right hand corner
// run them with activated journal channels to see what they look like
int main(int argc, char *argv[]) {
    // initialize the journal
    pyre::journal::init(argc, argv);
    pyre::journal::application("seq_transfer_tiles");
    // make a channel
    pyre::journal::debug_t channel("ampcor.correlators.seq");

    // the base dimension
    auto dim = 8;
    // the number of tile pairs
    auto pairs = 4;

    // the shape of the reference tiles
    seq_t::arena_shape_type refArenaShape { pairs, dim/4, dim/4 };
    // and their layout: 0-based, row major
    seq_t::arena_layout_type refArenaLayout { refArenaShape };

    // the shape of the secondary tiles, which includes their margin
    seq_t::arena_shape_type secArenaShape { pairs, dim/2, dim/2 };
    // the origin: chosen such that a zero shift corresponds to the tile with maximum
    // correlation having origin at (0,0)
    seq_t::arena_index_type secArenaOrigin { 0, -dim/8, -dim/8 };
    // the secondary tile layout
    seq_t::arena_layout_type secArenaLayout { secArenaShape, secArenaOrigin };

    // make a sequential worker with 4 pairs of tiles, trivial refinement and zoom
    seq_t seq(pairs, refArenaLayout, secArenaLayout, 1, 0, 1);

    // specify and open the rasters
    // shape
    slc_const_raster_t::shape_type rasterShape { dim, dim };
    // product spec
    slc_const_raster_t::spec_type spec { slc_const_raster_t::layout_type(rasterShape) };
    // open the sample reference raster in read-only mode
    slc_const_raster_t refRaster { spec, "slc_ref.dat" };
    // repeat for the secondary raster
    slc_const_raster_t secRaster { spec, "slc_sec.dat" };

    // we have four pairs of tiles to transfer
    for (auto i : {0,1}) {
        for (auto j : {0,1}) {
            // form the collation order
            auto tid = 2*i + j;
            // add the tiles
            seq.addTilePair(tid, tid,
                            refRaster.tile({i*dim/2 + dim/8, j*dim/2 + dim/8}, {dim/4,dim/4}),
                            secRaster.tile({i*dim/2,j*dim/2}, {dim/2,dim/2}));
        }
    }

    // all done
    return 0;
}


// end of file
