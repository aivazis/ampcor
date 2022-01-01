// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2022 all rights reserved


// support
#include <cassert>
// get the header
#include <ampcor/dom.h>
// the plan details
#include "plan.h"


// type aliases
using slc_raster_t = ampcor::dom::slc_raster_t;
using spec_t = slc_raster_t::spec_type;
using pixel_t = spec_t::pixel_type;
using value_t = spec_t::value_type;
using layout_t = slc_raster_t::layout_type;
using shape_t = slc_raster_t::shape_type;
using index_t = slc_raster_t::index_type;


// create the reference SLC raster:
//   . . . . . . . . . .
//   . x x x . . x x x .
//   . x x x . . x x x .
//   . x x x . . x x x .
//   . . . . . . . . . .
//   . . . . . . . . . .
//   . x x x . . x x x .
//   . x x x . . x x x .
//   . x x x . . x x x .
//   . . . . . . . . . .
//
int main(int argc, char *argv[]) {
    // initialize the journal
    pyre::journal::init(argc, argv);
    pyre::journal::application("slc_ref");
    // make a channel
    pyre::journal::debug_t channel("ampcor.correlators.slc");

    // the name of the product
    std::string name = "slc_ref.dat";
    // the shape of a seed tile: margin + data + margin
    shape_t tileShape = plan.seedShape + 2 * plan.seedMargin;
    // the plan grid shape tells me how many of these tiles to build
    shape_t gridShape = plan.gridShape;
    // so the product shape is
    shape_t slcShape { gridShape[0]*tileShape[0], gridShape[1]*tileShape[1] };
    // turn this into a layout
    layout_t slcLayout { slcShape };
    // build the product specification
    spec_t spec { slcLayout };

    // make the product
    slc_raster_t slc { spec, name, spec.cells() };

    // now, fill with data
    for (auto idx : layout_t(gridShape)) {
        // the origin of the non-zero part
        index_t tileOrigin {
            idx[0] * tileShape[0] + plan.seedMargin[0],
            idx[1] * tileShape[1] + plan.seedMargin[1]
        };
        // make a tile out of it
        auto tile = slc.box(tileOrigin, plan.seedShape);

        // get the central pixel of the tile
        auto center = tileOrigin + plan.seedShape/2;

        // loop over each pixel in the tile
        for (const auto & tdx : tile.layout()) {
            // compute the offset from the center tile
            auto offset = tdx - center;
            // compute the distance from the center
            float r2 = offset[0]*offset[0] + offset[1]*offset[1];

            // make an amplitude
            float amp = std::exp(-r2/2);
            // and a phase
            float phi = std::atan2(offset[1], offset[0]);

            // form the value
            pixel_t value { amp*std::cos(phi), amp*std::sin(phi) };
            // and set the corresponding cell
            slc[tdx] = value;
        }
    }

    // show me
    channel << "reference:" << pyre::journal::newline;
    for (auto i=0; i<slcShape[0]; ++i) {
        for (auto j=0; j<slcShape[1]; ++j) {
            channel << "  " << std::setw(19) << std::setprecision(2) << slc[{i,j}];
        }
        channel << pyre::journal::newline;
    }
    channel << pyre::journal::endl;

    // all done
    return 0;
}


// end of file
