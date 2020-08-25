// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


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
    pyre::journal::application("coarse_sat");
    // make a channel
    pyre::journal::debug_t channel("ampcor.correlators.sat");

    // the number of pairs
    auto pairs = plan.pairs;
    // the base dimension
    auto dim = plan.dim;

    // the name of the product with the secondary arena
    std::string arenaName = "coarse_sec.dat";
    // the name of the product with the SAT table
    std::string satName = "coarse_sat.dat";

    // the shape of a reference tile in its arena
    shape_t refShape { 1, dim/4, dim/4 };
    // the shape of a secondary tile in its arena
    shape_t secShape { 1, dim/2, dim/2 };
    // the shape of all possible placements of a {ref} tile within a {sec} tile
    shape_t plcShape {
        pairs, secShape[1] - refShape[1] + 1, secShape[2] - refShape[2] + 1 };

    // a useful index shift
    index_t deltaUL = { 0, -1, -1 };

    // arena: the origin
    index_t arenaOrigin { 0, -dim/8, -dim/8 };
    // the shape
    shape_t arenaShape { pairs, dim/2, dim/2 };
    // build the layout
    layout_t arenaLayout { arenaShape, arenaOrigin };
    // the product specification
    spec_t arenaSpec { arenaLayout };

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
    // verify: slide chips around the arena and verify that the sum of its elements can be
    // obtained by using the sat table

    // for each index that describes the origin of a placement of the chip with the secondary
    // tile, we need the four indices in the SAT that are used in the computation of the sum
    // project the chip shape on the tile axes
    index_t chip_1 = { 0, refShape[1], 0 };
    index_t chip_2 = { 0, 0, refShape[2] };
    // compute them as shift relative to the chip origin
    index_t deltaUR = deltaUL + chip_2;
    index_t deltaLL = deltaUL + chip_1;
    index_t deltaLR = deltaUL + chip_1 + chip_2;

    // initializer of the accumulator
    pixel_t zero = 0;
#if VERIFY
    // a small float
    auto epsilon = std::numeric_limits<pixel_t>::epsilon();
#endif

    // go through all possible placements of ref tiles within secondary tiles in the arena
    for (auto idx : arenaLayout.box(arenaOrigin, plcShape)) {
        // form a ref chip at this location
        auto chip = arena.box(idx, refShape);
        // compute the sum of its elements directly
        auto expected = std::accumulate(chip.begin(), chip.end(), zero);
        // get the sum using the SAT
        auto deduced = sat[idx+deltaUL] + sat[idx+deltaLR] - sat[idx+deltaUR] - sat[idx+deltaLL];

        // show me
        channel
            << "sum[" << idx << "]: "
            << pyre::journal::newline
            // the direct computation
            << "    expected: " << expected
            << pyre::journal::newline
            // form the SAT
            << "     deduced: " << deduced
            << pyre::journal::newline
            // here is the expression
            << "              = sat[" << idx+deltaUL << "]"
            << " + sat[" << idx+deltaLR << "]"
            << " - sat[" << idx+deltaUR << "]"
            << " - sat[" << idx+deltaLL << "]"
            << pyre::journal::newline
            // and the values
            << "              = " << sat[idx+deltaUL]
            << " + " << sat[idx+deltaLR]
            << " - " << sat[idx+deltaUR]
            << " - " << sat[idx+deltaLL]
            << pyre::journal::newline;
#if VERIFY
        // all this, and still not good enough
        // verify; tricky with floating point numbers...
        // use the mean actual value as an indicator of the dynamic range of the values
        auto mean = expected / refShape.cells();
        // for small numbers
        if (std::abs(expected) < 1 or std::abs(deduced) < 1) {
            // verify that they are within a few epsilon
            assert(( std::abs(expected-deduced) <= mean*epsilon ));
        } else {
            // otherwise, get the larger one
            auto base = std::max(std::abs(expected), std::abs(deduced));
            // compute the relative error
            auto delta = std::abs(expected-deduced) / base;
            // and check that it is a few epsilon
            assert(( delta <= mean*epsilon ));
        }
#endif
    }

    // flush
    channel << pyre::journal::endl(__HERE__);
    // all done
    return 0;
}


// end of file
