// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2021 all rights reserved


// support
#include <cassert>
// get the header
#include <ampcor/dom.h>
// the plan details
#include "plan.h"


// type aliases
using arena_const_raster_t = ampcor::dom::arena_const_raster_t<float>;
using spec_t = arena_const_raster_t::spec_type;
using pixel_t = spec_t::pixel_type;
using layout_t = arena_const_raster_t::layout_type;
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

    // to make the shape, combine the number of pairs
    spec_t::id_layout_type::shape_type pairsShape { plan.gridShape.cells() };
    // with the shape of a reference tile
    shape_t shape = pairsShape * plan.seedShape;
    // the origin of the reference arena
    index_t origin { 0, 0, 0 };
    // make a layout
    layout_t layout { shape, origin };

    // build the product specification
    spec_t spec { layout };

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
                channel << "  " << std::setprecision(2) << std::setw(7) << arena[idx];
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
