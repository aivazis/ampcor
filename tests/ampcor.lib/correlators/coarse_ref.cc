// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// support
#include <cassert>
// get the header
#include <ampcor/dom.h>


// type aliases
using arena_const_raster_t = ampcor::dom::arena_const_raster_t;
using spec_t = arena_const_raster_t::spec_type;
using pixel_t = spec_t::pixel_type;
using shape_t = arena_const_raster_t::shape_type;
using index_t = arena_const_raster_t::index_type;


// load the product that has the coarse arena with the reference tiles and dump its contents
int main(int argc, char *argv[]) {
    // initialize the journal
    pyre::journal::init(argc, argv);
    pyre::journal::application("arena_ref_coarse");
    // make a channel
    pyre::journal::debug_t channel("ampcor.correlators.arena");

    // the name of the product
    std::string name = "coarse_ref.dat";
    // the number of pairs
    auto pairs = 4;
    // the base dimension
    auto dim = 8;

    // the origin
    index_t origin { 0, 0, 0 };
    // make a shape
    shape_t shape { pairs, dim/4, dim/4 };
    // build the product specification
    spec_t spec { spec_t::layout_type(shape, origin) };

    // build the product
    arena_const_raster_t arena { spec, name };

    // show me
    channel << "coarse reference arena:" << pyre::journal::newline;
    // go through the tiles
    for (auto tid = origin[0]; tid < origin[0] + shape[0]; ++tid) {
        // show me the tile number
        channel << "tile " << tid << pyre::journal::newline;
        // and the contents
        for (auto i = origin[1]; i < origin[1] + shape[1]; ++i) {
            for (auto j = origin[2]; j < origin[2] + shape[2]; ++j) {
                // form the index
                index_t idx { tid, i, j };
                // show me
                channel << "  " << std::setw(7) << arena[idx];
            }
            channel << pyre::journal::newline;
        }
    }
    // flush
    channel << pyre::journal::endl(__HERE__);

    // all done
    return 0;
}


// end of file
