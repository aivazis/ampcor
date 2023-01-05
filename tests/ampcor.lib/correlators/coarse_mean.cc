// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2023 all rights reserved


// support
#include <cassert>
#include <algorithm>
// get the header
#include <ampcor/dom.h>
// the plan details
#include "plan.h"


// type aliases
using arena_const_raster_t = ampcor::dom::arena_const_raster_t<float>;
using spec_t = arena_const_raster_t::spec_type;
using pixel_t = spec_t::pixel_type;
using index_t = arena_const_raster_t::index_type;
using shape_t = arena_const_raster_t::shape_type;
using layout_t = arena_const_raster_t::layout_type;


// load the product that has the coarse arena with the reference tiles and dump its contents
int main(int argc, char *argv[]) {
    // initialize the journal
    pyre::journal::init(argc, argv);
    pyre::journal::application("coarse_mean");
    // make a channel
    pyre::journal::debug_t channel("ampcor.correlators.sec.mean");

    // the pair portion of the arena origin
    spec_t::id_layout_type::index_type none { 0 };
    // the number of pairs as an index part
    spec_t::id_layout_type::index_type pairs { plan.gridShape.cells() };
    // term that help form all possible placements of ref tiles in the secondary window
    ampcor::dom::slc_raster_t::shape_type one { 1, 1 };

    // arena: the name of the product
    std::string arenaName = "coarse_mean.dat";
    // the origin
    index_t arenaOrigin = none * (-1 * plan.seedMargin);
    // the shape: all possible placements of a {ref} tile within a {sec} tile
    shape_t arenaShape = pairs * (2 * plan.seedMargin + one);
    // build the layout
    layout_t arenaLayout { arenaShape, arenaOrigin };
    // the product specification
    spec_t arenaSpec { arenaLayout };

    // build the product
    arena_const_raster_t arena { arenaSpec, arenaName };

    // go through the tiles
    for (auto tid = 0; tid < pairs[0]; ++tid) {
        // show me the tile number
        channel << "tile " << tid << pyre::journal::newline;
        // and the contents
        for (auto i = arenaOrigin[1]; i < arenaOrigin[1] + arenaShape[1]; ++i) {
            for (auto j = arenaOrigin[2]; j < arenaOrigin[2] + arenaShape[2]; ++j) {
                // form the index
                index_t idx { tid, i, j };
                // show me
                channel << "  " << std::setprecision(3) << std::setw(10) << arena[idx];
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
