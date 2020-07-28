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
using offsets_t = ampcor::dom::offsets_t;


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
    offsets_t::shape_type shape { 2, 120, 40};
    // build the product specification
    offsets_t::spec_type spec { shape };

    // make the product; supplying the grid capacity is the signal to create a new one rather
    // than map over an existing product
    offsets_t offsets { spec, name, spec.cells() };

    // go through it
    for (const auto & idx : offsets.layout()) {
        // make a value out of the raster address
        offsets_t::pixel_type value { idx[1], idx[2] };
        // and store it
        offsets[idx] = value;
    }

    // once more
    for (const auto & idx : offsets.layout()) {
        // convert the index into a pixel
        offsets_t::pixel_type expected { idx[1], idx[2] };
        // verify
        assert(( offsets[idx] == expected ));
    }

    // all done
    return 0;
}


// end of file
