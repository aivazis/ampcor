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
using slc_t = ampcor::dom::slc_t;
using spec_t = slc_t::spec_type;
using pixel_t = spec_t::pixel_type;
using value_t = spec_t::value_type;
using shape_t = slc_t::shape_type;
using index_t = slc_t::index_type;


// create the secondary SLC raster:
//   . . . . . . . .
//   . . . . . . . .
//   . . x x . . x x
//   . . x x . . x x
//   . . . . . . . .
//   . . . . . . . .
//   . . x x . . x x
//   . . x x . . x x
//
int main(int argc, char *argv[]) {
    // initialize the journal
    pyre::journal::init(argc, argv);
    pyre::journal::application("slc_set");
    // make a channel
    pyre::journal::debug_t channel("ampcor.correlators.slc");

    // the name of the product
    std::string name = "slc_sec.dat";
    // the base dimension
    size_t dim = 256;
    // make a shape
    shape_t shape { dim, dim };
    // build the product specification
    spec_t spec { shape };

    // make the product
    slc_t slc { spec, name, spec.cells() };

    // form a 2x2 grid of tiles
    for (size_t i : {0,1}) {
        for (size_t j : {0,1}) {
            // form the base of the tile
            index_t base { i*dim/2 + dim/4, j*dim/2 + dim/4 };
            // form the shape of the tile
            shape_t shp { dim/4, dim/4 };
            // make the tile
            auto tile = slc.tile(base, shp);
            // now, loop over each tile index space
            for (const auto & idx : tile.layout()) {
                // form the value
                pixel_t value = static_cast<value_t>(idx[0]) + static_cast<value_t>(idx[1])*1if;
                // and set the corresponding cell
                slc[idx] = value;
            }
        }
    }

    // show me
    channel << "base:" << pyre::journal::newline;
    for (size_t i=0; i<shape[0]; ++i) {
        for (size_t j=0; j<shape[1]; ++j) {
            channel << "  " << std::setw(2) << slc[{i,j}];
        }
        channel << pyre::journal::newline;
    }
    channel << pyre::journal::endl;

    // all done
    return 0;
}


// end of file
