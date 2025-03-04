// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2025 all rights reserved


// support
#include <cassert>
// get the header
#include <ampcor/dom.h>
// the plan details
#include "plan.h"


// type aliases
using slc_raster_t = ampcor::dom::slc_raster_t;
using slc_const_raster_t = ampcor::dom::slc_const_raster_t;
using spec_t = slc_raster_t::spec_type;
using pixel_t = spec_t::pixel_type;
using value_t = spec_t::value_type;
using layout_t = slc_raster_t::layout_type;
using shape_t = slc_raster_t::shape_type;
using index_t = slc_raster_t::index_type;


// the secondary SLC raster, in block form:
//   X X X . . . . X X X
//   X X X . . . . X X X
//   X X X . . . . X X X
//   . . . . . . . . . .
//   . . . . . . . . . .
//   . . . . . . . . . .
//   . . . . . . . . . .
//   X X X . . . . X X X
//   X X X . . . . X X X
//   X X X . . . . X X X
// the data come from reading the reference slc
int main(int argc, char *argv[]) {
    // initialize the journal
    pyre::journal::init(argc, argv);
    pyre::journal::application("slc_sec");
    // make a channel
    pyre::journal::debug_t channel("ampcor.correlators.slc");

    // the names of the products
    std::string refName = "slc_ref.dat";
    std::string secName = "slc_sec.dat";

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

    // get the reference raster
    slc_const_raster_t ref { spec, refName };
    // make the product
    slc_raster_t sec { spec, secName, spec.cells() };

    // now, fill with data
    for (auto idx : layout_t(gridShape)) {
        // the source origin
        index_t refOrigin {
            idx[0] * tileShape[0] + plan.seedMargin[0],
            idx[1] * tileShape[1] + plan.seedMargin[1]
        };
        // the destination origin
        index_t secOrigin {
            idx[0] * (2*plan.seedMargin[0] + tileShape[0]),
            idx[1] * (2*plan.seedMargin[1] + tileShape[1])
        };

        // identify the reference data
        auto src = ref.box(refOrigin, plan.seedShape);
        // identify the destination
        auto dst = sec.box(secOrigin, plan.seedShape);
        // copy the data
        std::copy(src.begin(), src.end(), dst.begin());
    }

    // show me
    channel << "secondary:" << pyre::journal::newline;
    for (auto i=0; i<slcShape[0]; ++i) {
        for (auto j=0; j<slcShape[1]; ++j) {
            channel << "  " << std::setw(19) << std::setprecision(2) << sec[{i,j}];
        }
        channel << pyre::journal::newline;
    }
    channel << pyre::journal::endl;

    // all done
    return 0;
}


// end of file
