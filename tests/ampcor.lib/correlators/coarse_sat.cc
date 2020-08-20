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
using index_t = arena_const_raster_t::index_type;
using shape_t = arena_const_raster_t::shape_type;
using layout_t = arena_const_raster_t::layout_type;


// load the product that has the coarse arena with the reference tiles and dump its contents
int main(int argc, char *argv[]) {
    // initialize the journal
    pyre::journal::init(argc, argv);
    pyre::journal::application("coarse_sat");
    // make a channel
    pyre::journal::debug_t channel("ampcor.correlators.arena");

    // the number of pairs
    auto pairs = 4;
    // the base dimension
    auto dim = 8;
    // a useful index shift
    index_t deltaUL = { 0, -1, -1 };

    // the name of the product with the secondary arena
    std::string arenaName = "coarse_sec.dat";
    // the name of the product with the SAT table
    std::string satName = "coarse_sat.dat";

    // arena: the origin
    index_t arenaOrigin { 0, -dim/8, -dim/8 };
    // the shape
    shape_t arenaShape { pairs, dim/2, dim/2 };
    // the product specification
    spec_t arenaSpec { spec_t::layout_type(arenaShape, arenaOrigin) };

    // SAT: the origin
    index_t satOrigin = arenaOrigin + deltaUL;
    // the shape
    shape_t satShape { pairs, dim/2+1, dim/2+1 };
    // the product specification
    spec_t satSpec { spec_t::layout_type(satShape, satOrigin) };

    // build the products
    arena_const_raster_t arena { arenaSpec, arenaName };
    arena_const_raster_t sat { satSpec, satName };

    // go through the tiles
    for (auto tid = arenaOrigin[0]; tid < arenaOrigin[0] + arenaShape[0]; ++tid) {
        // show me the tile number
        channel << "tile " << tid << pyre::journal::newline;
        // and the contents
        for (auto i = arenaOrigin[1]; i < arenaOrigin[1] + arenaShape[1]; ++i) {
            for (auto j = arenaOrigin[2]; j < arenaOrigin[2] + arenaShape[2]; ++j) {
                // form the index
                index_t idx { tid, i, j };
                // show me
                channel << "  " << std::setw(7) << arena[idx];
            }
            channel << pyre::journal::newline;
        }

        // SAT
        channel << "sum area table:" << pyre::journal::newline;
        // the contents
        for (auto i = satOrigin[1]; i < satOrigin[1] + satShape[1]; ++i) {
            for (auto j = satOrigin[2]; j < satOrigin[2] + satShape[2]; ++j) {
                // form the index
                index_t idx { tid, i, j };
                // show me
                channel << "  " << std::setw(7) << sat[idx];
            }
            channel << pyre::journal::newline;
        }
    }
    // flush
    channel << pyre::journal::endl(__HERE__);

    // verify: slide chips around the arena and verify that the sum of its elements can be
    // obtained by using the sat table
    // make a shape that describes our sliding chip
    shape_t chipShape { 1, dim/4, dim/4 };

    // for each index that describes the placement of the chip with the secondary tile, we need
    // the four indices in the SAT that are used in the computation of the sum
    // project the chip shape on the tile axes
    index_t chip_1 = { 0, chipShape[1], 0 };
    index_t chip_2 = { 0, 0, chipShape[2] };
    // compute them as shift relative to the chip origin
    index_t deltaUR = deltaUL + chip_2;
    index_t deltaLL = deltaUL + chip_1;
    index_t deltaLR = deltaUL + chip_1 + chip_2;

    // we do some floating point comparisons, so...
    pixel_t zero = 0;

    // go through all the tiles
    for (auto tid = arenaOrigin[0]; tid < arenaOrigin[0] + arenaShape[0]; ++tid) {
        // build a layout that will give me all the possible placements of a chip inside the
        // arena tile
        channel << "tile #" << tid << pyre::journal::newline;
        // the origin
        index_t plOrigin { tid, arenaOrigin[1], arenaOrigin[2] };
        // the shape
        shape_t plShape { tid, arenaShape[1]-chipShape[1]+1, arenaShape[2]-chipShape[2]+1 };
        // the layout
        layout_t pl { plShape , plOrigin };

        // go through all placements
        for (auto chipOrigin : pl ) {
            // form the tile
            auto tile = arena.box(chipOrigin, chipShape);
            // compute the sum by visiting the tile slots
            auto expected = std::accumulate(tile.begin(), tile.end(), zero);
            // compute the sum by indexing the SAT
            auto actual =
                sat[chipOrigin+deltaUL] + sat[chipOrigin+deltaLR]
                - sat[chipOrigin+deltaUR] - sat[chipOrigin+deltaLL];

            // show me; this is a good example of the need for level of detail in channel output
            channel
                << chipOrigin << ": " << pyre::journal::newline
                << "  expected = " << expected << pyre::journal::newline
                << "  actual = " << actual
                << " = " << sat[chipOrigin+deltaUL]
                << " + " << sat[chipOrigin+deltaLR]
                << " - " << sat[chipOrigin+deltaUR]
                << " - " << sat[chipOrigin+deltaLL]
                << pyre::journal::newline;
        }

        // flush
        channel << pyre::journal::endl(__HERE__);
    }

    // all done
    return 0;
}


// end of file
