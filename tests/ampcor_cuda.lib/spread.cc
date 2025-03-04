// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2025 all rights reserved
//

// configuration
#include <portinfo>
// ampcor
#include <ampcor/correlators.h>
#include <ampcor_cuda/correlators.h>


// type aliases
using value_t = float;
using complex_t = std::complex<value_t>;

// on the host
using arena_type = ampcor::dom::arena_raster_t<complex_t>;
using arena_layout_type = arena_type::layout_type;
using arena_shape_type = arena_type::shape_type;
using arena_index_type = arena_type::index_type;
// on the device
using dev_arena_type = ampcor::cuda::correlators::devarena_raster_t<complex_t>;


// test driver
int test() {
    // make a channel
    pyre::journal::info_t channel("ampcor.spread");

    // the number of pairs in the arena
    auto pairs = 2;
    // and the refinement factor
    auto refine = 2;

    // the narrow tiles
    auto narrowRows = 7;
    auto narrowCols = 5;
    // the refined tiles
    auto refinedRows = refine * narrowRows;
    auto refinedCols = refine * narrowCols;

    // the arena origin
    arena_index_type origin { 0, 0, 0 };
    // the shape of the narrow tiles
    arena_shape_type narrowTileShape { pairs, narrowRows, narrowCols };
    // use it describe the shape of the narrow tiles
    arena_layout_type narrowLayout { narrowTileShape, origin };


    // the shape of the refined tiles
    arena_shape_type refinedTileShape { pairs, refinedRows, refinedCols };
    // use it to assemble the arena layout
    arena_layout_type layout { refinedTileShape, origin };

    // make an arena
    dev_arena_type arena { layout, layout.cells() };
    // initialize it
    for (auto idx : narrowLayout) {
        // get the offset
        value_t offset = narrowLayout.offset(idx);
        // use it to make a complex number
        complex_t value { offset, offset };
        // and store it in the arena
        arena[idx] = value;
    }

    // show me before
    for (auto pid = 0; pid < pairs; ++pid) {
        // display the pair id
        channel
            << "pid " << pid << ":"
            << pyre::journal::newline;
        // the tile contents
        for (auto row = 0; row < refinedRows; ++row) {
            for (auto col = 0; col < refinedCols; ++col) {
                // turn all this into an index
                arena_index_type idx { pid, row, col };
                // get the value and print it
                channel << " " << std::setw(7) << arena[idx];
            }
            channel << pyre::journal::newline;
        }
    }
    // flush
    channel << pyre::journal::endl(__HERE__);

    // spread the spectrum
    ampcor::cuda::kernels::spread(arena.data()->data(),
                                  pairs, refinedRows, refinedCols,
                                  narrowRows, narrowCols);

    // show me after
    for (auto pid = 0; pid < pairs; ++pid) {
        // display the pair id
        channel
            << "pid " << pid << ":"
            << pyre::journal::newline;
        // the tile contents
        for (auto row = 0; row < refinedRows; ++row) {
            for (auto col = 0; col < refinedCols; ++col) {
                // turn all this into an index
                arena_index_type idx { pid, row, col };
                // get the value and print it
                channel << " " << std::setw(7) << arena[idx];
            }
            channel << pyre::journal::newline;
        }
    }
    // flush
    channel << pyre::journal::endl(__HERE__);

    // all done
    return 0;
}


// entry point
int main()
{
    // access the top level info channel
    pyre::journal::info_t channel("ampcor");
    // so we can quiet it down
    channel.deactivate();

    // pick a device
    cudaSetDevice(0);
    // test
    auto status = test();
    // reset the device
    cudaDeviceReset();

    // all done
    return status;
}


// end of file
