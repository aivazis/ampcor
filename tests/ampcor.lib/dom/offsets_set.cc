// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// support
#include <cassert>
// get the header
#include <ampcor/dom.h>
// access the complex literals
using namespace std::complex_literals;


// type aliases
using offsets_raster_t = ampcor::dom::offsets_raster_t;


// create an offset map
int main(int argc, char *argv[]) {
    // initialize the journal
    pyre::journal::init(argc, argv);
    pyre::journal::application("offsets_set");
    // make a channel
    pyre::journal::debug_t channel("ampcor.dom.offsets");

    // the name of the product
    std::string name = "offsets.dat";
    // make a shape
    offsets_raster_t::shape_type shape { 120, 40 };
    // build the product specification
    offsets_raster_t::spec_type spec { offsets_raster_t::layout_type(shape) };

    // make the product; supplying the grid capacity is the signal to create a new one rather
    // than map over an existing product
    offsets_raster_t offsets { spec, name, spec.cells() };

    // go through it
    for (const auto & idx : offsets.layout()) {
        // make some values out of the raster address
        float x = idx[0];
        float y = idx[1];
        float dx = x / 2;
        float dy = x / 2;
        // build a cell
        offsets_raster_t::pixel_type value { {x, y}, {dx, dy}, 0, 0, 0 };
        // and store it
        offsets[idx] = value;
    }

    // once more
    for (const auto & idx : offsets.layout()) {
        // the expected values
        float x = idx[0];
        float y = idx[1];
        float dx = x / 2;
        float dy = x / 2;
        // build a cell
        offsets_raster_t::pixel_type expected { {x, y}, {dx, dy}, 0, 0, 0 };
        // get the raster value
        auto & value = offsets[idx];
        // verify
        assert(( value.ref == expected.ref ));
        assert(( value.shift == expected.shift ));
        assert(( value.confidence == expected.confidence ));
        assert(( value.snr == expected.snr ));
        assert(( value.covariance == expected.covariance ));
    }

    // all done
    return 0;
}


// end of file
