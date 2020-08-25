// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


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
//   X X . . . . X X
//   X X . . . . X X
//   . . . . . . . .
//   . . . . . . . .
//   . . . . . . . .
//   . . . . . . . .
//   X X . . . . x x
//   X X . . . . x x
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

    // the base dimension
    auto dim = plan.dim;
    // make a shape
    shape_t rasterShape { dim, dim };
    // build the product specification
    spec_t spec { layout_t(rasterShape) };

    // get the reference raster
    slc_const_raster_t ref { spec, refName };
    // make the product
    slc_raster_t sec { spec, secName, spec.cells() };

    // form a 2x2 grid of tiles
    for (auto i : {0,1}) {
        for (auto j : {0,1}) {
            // identify the reference data
            auto src = ref.box({i*dim/2+dim/8, j*dim/2+dim/8}, {dim/4, dim/4});
            // identify the destination
            auto dst = sec.box({3*dim*i/4, 3*dim*j/4}, {dim/4, dim/4});
            // copy the data
            std::copy(src.begin(), src.end(), dst.begin());
        }
    }

    // show me
    channel << "secondary:" << pyre::journal::newline;
    for (auto i=0; i<rasterShape[0]; ++i) {
        for (auto j=0; j<rasterShape[1]; ++j) {
            channel << "  " << std::setw(7) << sec[{i,j}];
        }
        channel << pyre::journal::newline;
    }
    channel << pyre::journal::endl;

    // all done
    return 0;
}


// end of file
