// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2023 all rights reserved


// support
#include <cassert>
// get the headers
#include <ampcor/dom.h>
#include <ampcor/correlators.h>
// the plan details
#include "plan.h"


// type aliases
// products
using slc_const_raster_t = ampcor::dom::slc_const_raster_t;
using offsets_raster_t = ampcor::dom::offsets_raster_t;
// the correlator
using seq_t = ampcor::correlators::sequential_t<slc_const_raster_t, offsets_raster_t>;


using slc_const_raster_t = ampcor::dom::slc_const_raster_t;
using offsets_raster_t = ampcor::dom::offsets_raster_t;
// make a sequential worker and add tiles to its arena
// - the tiles come from the output of the {slc_ref} and {slc_sec} test cases
// run them with activated journal channels to see what they look like
int main(int argc, char *argv[]) {
    // initialize the journal
    pyre::journal::init(argc, argv);
    pyre::journal::application("seq_adjust");
    // make a channel
    pyre::journal::debug_t channel("ampcor.correlators.seq");

    // silence the {sequential_t} info channel
    // pyre::journal::info_t("ampcor.sequential").deactivate();

    // the shape of a seed tile: margin + data + margin
    slc_const_raster_t::shape_type tileShape = plan.seedShape + 2 * plan.seedMargin;
    // the plan grid shape tells me how many of these tiles to build
    slc_const_raster_t::shape_type gridShape = plan.gridShape;
    // so the product shape is
    slc_const_raster_t::shape_type slcShape {
        gridShape[0]*tileShape[0],
        gridShape[1]*tileShape[1]
    };
    // turn this into a layout
    slc_const_raster_t::layout_type slcLayout { slcShape };
    // build the product specification
    slc_const_raster_t::spec_type spec { slcLayout };

    // specify and open the input rasters
    slc_const_raster_t ref { spec, "slc_ref.dat" };
    slc_const_raster_t sec { spec, "slc_sec.dat" };

    // the output is a 2x2 grid
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
        // shift to its center
        auto ref = origin + tileShape / 2;
        // and save
        rec.ref = std::make_pair<float, float>(ref[0], ref[1]);
    }

    // make a sequential worker with 4 pairs of tiles, trivial refinement and zoom
    seq_t seq(0, ref, sec, offsets, plan.seedShape, tileShape,
              plan.refineFactor, plan.refineMargin, plan.zoomFactor);
    // estimate the offsets
    seq.adjust(offsets.layout());

    // all done
    return 0;
}


// end of file
