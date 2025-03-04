// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2025 all rights reserved


// support
#include <cassert>
// get the header
#include <ampcor/dom.h>
// access the complex literals
using namespace std::complex_literals;


// type aliases
using slc_const_raster_t = ampcor::dom::slc_const_raster_t;


// create an SLC
int main(int argc, char *argv[]) {
    // initialize the journal
    pyre::journal::init(argc, argv);
    pyre::journal::application("slc_get");
    // make a channel
    pyre::journal::debug_t channel("ampcor.dom.slc");

    // the name of the product
    std::string name = "slc.dat";
    // make a shape
    slc_const_raster_t::shape_type shape { 256, 256 };
    // build the product specification
    slc_const_raster_t::spec_type spec { slc_const_raster_t::layout_type(shape) };

    // open the product
    slc_const_raster_t slc { spec, name };

    // go through it
    for (const auto & idx : slc.layout()) {
        // convert the index into a complex float
        auto expected = static_cast<float>(idx[0]) + static_cast<float>(idx[1]) * 1if;
        // verify that the file contains what we expect
        assert(( slc[idx] == expected ));
    }

    // all done
    return 0;
}


// end of file
