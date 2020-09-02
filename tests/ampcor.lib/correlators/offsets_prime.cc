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
// the plan details
#include "plan.h"


// type aliases
using offsets_raster_t = ampcor::dom::offsets_raster_t;
using slc_const_raster_t = ampcor::dom::slc_const_raster_t;


// initialize the offsets product
int main(int argc, char *argv[]) {
    // initialize the journal
    pyre::journal::init(argc, argv);
    pyre::journal::application("offsets_new");
    // make a channel
    pyre::journal::debug_t channel("ampcor.correlators.offsets");

    // the shape of a seed tile: margin + data + margin
    auto tileShape = plan.seedShape + 2 * plan.seedMargin;

    // the output layout
    offsets_raster_t::layout_type offsetLayout { plan.gridShape };
    // build the spec of the output product
    offsets_raster_t::spec_type offsetSpec { offsetLayout };
    // open the output
    offsets_raster_t offsets { offsetSpec, "offsets.dat", offsetSpec.cells() };
    // prime it
    for (auto idx : offsets.layout()) {
        // get the record
        auto & rec = offsets[idx];
        // find the origin of this tile
        slc_const_raster_t::shape_type origin { idx[0] * tileShape[0], idx[1] * tileShape[1] };
        // shift to its center and save
        rec.ref = origin + tileShape / 2;

        // show me
        channel
            << "(" << idx << ") : (" << rec.ref << ") --> (" << rec.ref + rec.shift << ")"
            << pyre::journal::newline;
    }

    // flush
    channel << pyre::journal::endl(__HERE__);

    // all done
    return 0;
}


// end of file
