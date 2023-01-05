// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2023 all rights reserved
//

// configuration
#include <portinfo>
// external
#include <iostream>
// ampcor
#include <ampcor/correlators.h>
#include <ampcor_cuda/correlators.h>


// type aliases
using real_t = float;
using complex_t = std::complex<real_t>;

// on the host
using arena_type = ampcor::dom::arena_raster_t<complex_t>;
using arena_layout_type = arena_type::layout_type;
using arena_shape_type = arena_type::shape_type;
using arena_index_type = arena_type::index_type;
// on the device
// real
using dev_arena_type = ampcor::cuda::correlators::devarena_raster_t<real_t>;
// complex
using dev_carena_type = ampcor::cuda::correlators::devarena_raster_t<complex_t>;


// test driver
int test() {
    // make a channel for showing the partial results
    pyre::journal::debug_t channel("detect.data");
    // and another for displaying timing info
    pyre::journal::info_t tlog("detect.time");

    // the number of pairs in the arena
    auto pairs = 4096;
    // the tiles
    auto rows = 127;
    auto cols = 253;

    // the origin
    arena_index_type origin { 0, 0, 0 };
    // the shape
    arena_shape_type shape { pairs, rows, cols };
    // the layout
    arena_layout_type layout { shape, origin };

    // use the layout to make a complex arena
    dev_carena_type carena { layout, layout.cells() };

    // initialize it
    for (auto idx : layout) {
        // get the offset
        real_t offset = layout.offset(idx);
        // use it to make a complex number
        complex_t value { offset, offset };
        // and store it in the arena
        carena[idx] = value;
    }

    // if the user cares
    if (channel) {
        // show me the complex arena
        for (auto pid = 0; pid < pairs; ++pid) {
            // display the pair id
            channel
                << "pid " << pid << ":"
                << pyre::journal::newline;
            // the tile contents
            for (auto row = 0; row < rows; ++row) {
                for (auto col = 0; col < cols; ++col) {
                    // turn all this into an index
                    arena_index_type idx { pid, row, col };
                    // get the value and print it
                    channel << " " << std::setw(9) << carena[idx];
                }
                channel << pyre::journal::newline;
            }
        }
        // flush
        channel << pyre::journal::endl(__HERE__);
    }

    // make a real arena with the same layout
    dev_arena_type arena { layout, layout.cells() };

    // make a timer
    pyre::timers::wall_timer_t t("ampcor.detect");
    // start it
    t.start();
    // detect the tiles
    ampcor::cuda::kernels::detect(carena.data()->data(),
                                  pairs, rows, cols, arena.data()->data());
    // stop the timer
    t.stop();
    // show me
    tlog
        << "detected " << layout.cells() << " complex numbers in " << t.us() << "us"
        << " (" << layout.cells() / t.sec() << ")"
        << pyre::journal::endl(__HERE__);

    // if the user cares
    if (channel) {
        // show me the amplitude arena
        for (auto pid = 0; pid < pairs; ++pid) {
            // display the pair id
            channel
                << "pid " << pid << ":"
                << pyre::journal::newline;
            // the tile contents
            for (auto row = 0; row < rows; ++row) {
                for (auto col = 0; col < cols; ++col) {
                    // turn all this into an index
                    arena_index_type idx { pid, row, col };
                    // get the value and print it
                    channel
                        << " "
                        << std::setw(9) << std::setprecision(3) << std::fixed << std::right
                        << arena[idx];
                }
                channel << pyre::journal::newline;
            }
        }
        // flush
        channel << pyre::journal::endl(__HERE__);
    }


    // build a tolerance
    real_t eps = 5 * std::numeric_limits<real_t>::epsilon();
    // compare
    for (auto idx : layout) {
        // compute what we expect
        auto expected = std::abs(carena[idx]);
        // read what we got
        auto actual = arena[idx];

        // verify; it's tricky with real numbers...
        // if the values are close enough in absolute terms
        if (std::abs(expected - actual) <= eps * std::abs(expected)) {
            // move on
            continue;
        }

        // make a channel
        pyre::journal::error_t error("ampcor.detect");
        // and complain
        error
            << "while verifying the cell at {" << idx << "}: error:"
            << pyre::journal::newline
            << " expected: " << expected
            << ", got: " << actual
            << pyre::journal::endl(__HERE__);
    }

    // all done
    return 0;
}


// entry point
int main(int argc, char *argv[])
{
    // get the timer channel
    pyre::journal::info_t tlog("detect.time");
    // so we can quiet it down
    tlog.deactivate();

    // the device id
    int did = 0;
    // if the user supplied a command line argument
    if (argc > 1) {
        // interpret the argument as the device id
        did = std::atoi(argv[1]);
    }

    // pick a device
    cudaSetDevice(did);
    // test
    auto status = test();
    // reset the device
    cudaDeviceReset();

    // all done
    return status;
}


// end of file
