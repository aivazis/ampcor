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
    pyre::journal::application("seq_sec");
    // make a channel
    pyre::journal::debug_t channel("ampcor.correlators.seq");

    // the base dimension
    size_t dim = 8;

    // the shape of the secondary raster
    slc_t::shape_type rasterShape { dim, dim };
    // the product spec
    slc_t::spec_type spec { rasterShape };
    // open the sample secondary raster in read-only mode
    slc_t slc { spec, "slc_sec.dat" };

    // show me the raster
    channel << "slc:" << pyre::journal::newline;
    for (size_t i=0; i<rasterShape[0]; ++i) {
        for (size_t j=0; j<rasterShape[1]; ++j) {
            channel << "  " << std::setw(2) << slc[{i,j}];
        }
        channel << pyre::journal::newline;
    }
    channel << pyre::journal::endl;


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

    // make a sequential worker with 4 pairs of tiles, trivial refinement and zoom
    seq_t seq(4, refLayout, secLayout, 1, 0, 1);
    // pick a value
    slc_t::spec_type::pixel_type value = 4.f + 42.if;
    // zero out its coarse arena
    seq.fillCoarseArena(value);

    // pick a spot somewhere in the middle of the secondary raster: the {1,0} tile
    slc_t::index_type i_10 { dim/2, 0 };
    // build the tile
    auto t_10 = slc.tile(i_10, secShape);

    channel << "t_10:" << pyre::journal::newline;
    for (auto i : t_10.layout()) {
        channel << "  t_10[" << i << "] = " << t_10[i] << pyre::journal::newline;
    }
    channel << pyre::journal::endl;

    // add it to the arena
    seq.addSecondaryTile(2, t_10);

    // verify
    // get the arena
    auto arena = seq.coarseArena();
    // its memory footprint
    auto bytes = seq.coarseArenaBytes();
    // and its stride to skip over one (ref,sec) pair
    auto stride = seq.coarseArenaStride();

    // show me
    channel
        << "arena: " << bytes << " bytes at " << arena
        << pyre::journal::endl;

    // the first ref/sec pair should be untouched
    for (auto c = arena; c != arena+stride; ++c) {
        // i.e. equal to the initialization value
        assert(( *c == value ));
    }
    // ditto for the second ref/sec pair
    for (auto c = arena+stride; c != arena+2*stride; ++c) {
        // i.e. equal to the initialization value
        assert(( *c == value ));
    }

    // the third pair has zeros for the reference tile
    for (auto c = arena+2*stride; c != arena+2*stride+refShape.cells(); ++c) {
        // i.e. equal to the initialization value
        assert(( *c == value ));
    }
    // and the secondary tile
    assert(( std::equal(t_10.begin(), t_10.end(), arena+2*stride+refShape.cells()) ));

    // the fourth pair is untouched
    for (auto c = arena+3*stride; c != arena+4*stride; ++c) {
        // i.e. equal to the initialization value
        assert(( *c == value ));
    }

    // all done
    return 0;
}


// end of file
