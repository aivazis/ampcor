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
using seq_t = ampcor::correlators::sequential_t<>;


// make a sequential worker and add the reference tiles to its arena
int main(int argc, char *argv[]) {
    // initialize the journal
    pyre::journal::init(argc, argv);
    pyre::journal::application("seq_coarse_zero");
    // make a channel
    pyre::journal::debug_t channel("ampcor.correlators.seq");

    // the number of pairs
    auto pairs = 4;
    // the base tile dimension
    auto dim = 8;

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

    // make a sequential worker
    seq_t seq(pairs, refLayout, secLayout, 1, 0, 1);

    // all we know is the number of pairs
    assert(( seq.pairs() == pairs ));

    // all done
    return 0;
}


// end of file
