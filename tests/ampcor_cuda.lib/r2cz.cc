// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2022 all rights reserved
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
    pyre::journal::info_t channel("r2cz.data");
    // and another for displaying timing info
    pyre::journal::info_t tlog("r2cz.time");

    // the number of pairs in the arena
    auto pairs = 1;
    // the tiles
    auto rows = 2;
    auto cols = 2;
    // the zoom factor
    auto zoomFactor = 3;
    // hence
    auto zoomedRows = zoomFactor * rows;
    auto zoomedCols = zoomFactor * cols;

    // the origin
    arena_index_type origin { 0, 0, 0 };
    // the shape
    arena_shape_type shape { pairs, rows, cols };
    // the layout
    arena_layout_type layout { shape, origin };

    // make a real arena with this layout
    dev_arena_type arena { layout, layout.cells() };
    // and initialize it
    for (auto idx : layout) {
        // get the offset
        real_t offset = layout.offset(idx);
        // and store it in the arena as the value of the cell
        arena[idx] = offset;
    }

    // if the user cares
    if (channel) {
        // show me the real+narrow arena
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
                        << std::setw(7) << std::setprecision(3) << std::fixed << std::right
                        << arena[idx];
                }
                channel << pyre::journal::newline;
            }
        }
        // flush
        channel << pyre::journal::endl(__HERE__);
    }

    // compute the zoomed origin
    auto zoomedOrigin = zoomFactor * origin;
    // and the zoomed shape
    arena_shape_type zoomedShape { pairs, zoomedRows, zoomedCols };
    // make the zoomed layout
    arena_layout_type zoomedLayout { zoomedShape, zoomedOrigin };
    // use this to allocate a complex zoom arena
    dev_carena_type carena { zoomedLayout, zoomedLayout.cells() };

    // make the source region
    auto src = arena.box(origin, shape);
    // and the destination region
    auto dst = carena.box(zoomedOrigin, shape);

    // copy
    std::copy(arena.begin(), arena.end(), dst.begin());

    // if the user cares
    if (channel) {
        // show me the amplitude arena
        for (auto pid = 0; pid < pairs; ++pid) {
            // display the pair id
            channel
                << "pid " << pid << ":"
                << pyre::journal::newline;
            // the tile contents
            for (auto row = 0; row < zoomedRows; ++row) {
                for (auto col = 0; col < zoomedCols; ++col) {
                    // turn all this into an index
                    arena_index_type idx { pid, row, col };
                    // get the value and print it
                    channel
                        << " "
                        << std::setw(9) << std::setprecision(3) << std::fixed << std::right
                        << carena[idx];
                }
                channel << pyre::journal::newline;
            }
        }
        // flush
        channel << pyre::journal::endl(__HERE__);
    }


#if 0
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
#endif

    // all done
    return 0;
}


// entry point
int main(int argc, char *argv[])
{
    // get the timer channel
    pyre::journal::info_t tlog("r2cz.time");
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
