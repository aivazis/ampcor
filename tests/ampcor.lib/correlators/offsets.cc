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
using raster_t = ampcor::dom::offsets_const_raster_t;
using spec_t = raster_t::spec_type;
using pixel_t = spec_t::pixel_type;
using layout_t = raster_t::layout_type;
using shape_t = raster_t::shape_type;
using index_t = raster_t::index_type;


// dump the contents of the {offsets} map
int main(int argc, char *argv[]) {
    // initialize the journal
    pyre::journal::init(argc, argv);
    pyre::journal::application("offsets");
    // make a channel
    pyre::journal::info_t channel("ampcor.correlators.offsets");

    // the name of the product
    std::string name = "offsets.dat";
    // build the product layout
    layout_t layout { plan.gridShape };
    // build the product specification
    spec_t spec { layout };
    // make the product
    raster_t offsets { spec, name };

    // go through it
    for (auto idx : offsets.layout()) {
        // get the record
        auto rec = offsets[idx];
        // get the source pixel
        auto ref = rec.ref;
        // form the destination pixel
        auto sec = ref + rec.shift;
        // get the confidence
        auto conf = rec.confidence;

        // show me
        channel
            << "(" << idx << "): "
            << "(" << ref << ") --> (" << sec << "), conf="
            << std::setprecision(3) << conf
            << pyre::journal::newline;
    }

    // flush
    channel << pyre::journal::endl(__HERE__);

    // all done
    return 0;
}


// end of file
