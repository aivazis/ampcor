// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2023 all rights reserved
//

// configuration
#include <portinfo>
// cuda
#include <cuda_runtime.h>
#include <cufft.h>
// support
#include <pyre/grid.h>
#include <pyre/journal.h>
#include <pyre/timers.h>
// ampcor
#include <ampcor_cuda/correlators.h>

// type aliases
// my value type
using value_t = float;
// the pixel type
using pixel_t = std::complex<value_t>;
// my raster type
using slc_t = pyre::grid::simple_t<2, pixel_t>;
// the correlator
using correlator_t = ampcor::cuda::correlators::sequential_t<slc_t>;

// driver
int main() {
    // make a timer
    pyre::timer_t timer("ampcor.cuda.sanity");
    // make a channel for reporting the timings
    pyre::journal::debug_t tlog("ampcor.cuda.tlog");

    // make a channel for logging progress
    pyre::journal::debug_t channel("ampcor.cuda");
    // sign in
    channel
        << pyre::journal::at(__HERE__)
        << "test: adding reference and secondary tile pairs to the correlator"
        << pyre::journal::endl;

    // the reference tile extent
    int refDim = 128;
    // the margin around the reference tile
    int margin = 32;
    // therefore, the secondary tile extent
    int secDim = refDim + 2*margin;
    // the number of possible placements of the reference tile within the secondary tile
    int placements = 2*margin + 1;
    // the number of pairs
    slc_t::size_type pairs = placements*placements;

    // the number of cells in a reference tile
    slc_t::size_type refCells = refDim * refDim;
    // the number of cells in a secondary tile
    slc_t::size_type secCells = secDim * secDim;
    // the number of cells per pair
    slc_t::size_type cellsPerPair = refCells + secCells;

    // the reference shape
    slc_t::shape_type refShape = {refDim, refDim};
    // the search window shape
    slc_t::shape_type secShape = {secDim, secDim};

    // the reference layout with the given shape and default packing
    slc_t::layout_type refLayout = { refShape };
    // the search window layout with the given shape and default packing
    slc_t::layout_type secLayout = { secShape };

    // start the clock
    timer.reset().start();
    // make a correlator
    correlator_t c(pairs, refLayout, secLayout);
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "instantiating the manager: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // build reference tiles
    for (auto i=0; i<placements; ++i) {
        for (auto j=0; j<placements; ++j) {
            // compute the pair id
            int pid = i*placements + j;

            // make a reference raster
            slc_t ref(refLayout);
            // fill it with ones
            std::fill(ref.view().begin(), ref.view().end(), pid);

            // make a secondary tile
            slc_t sec(secLayout);
            // fill it with zeroes
            std::fill(sec.view().begin(), sec.view().end(), 0);
            // make a slice
            slc_t::slice_type slice = sec.layout().slice({i,j}, {i+refDim, j+refDim});
            // make a view of the sec tile over this slice
            slc_t::view_type view = sec.view(slice);
            // fill it with ones
            std::copy(ref.view().begin(), ref.view().end(), view.begin());

            // show me
            channel << "sec[" << i << "," << j << "]:" << pyre::journal::newline;
            for (auto idx=0; idx<secDim; ++idx) {
                for (auto jdx=0; jdx<secDim; ++jdx) {
                    channel << sec[{idx, jdx}] << " ";
                }
                channel << pyre::journal::newline;
            }
            channel << pyre::journal::endl;

            // add this pair to the correlator
            c.addReferenceTile(pid, ref.constview());
            c.addSecondaryTile(pid, sec.constview());
        }
    }
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "creating reference dataset: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // verify
    // start the clock
    timer.reset().start();
    // get the arena
    auto arena = c.arena();
    // go through all pairs
    for (auto i=0; i<placements; ++i) {
        for (auto j=0; i<placements; ++i) {
            // compute the pair id
            auto pid = i*placements + j;
            // get the reference raster
            auto ref = arena + pid*cellsPerPair;
            // verify its contents
            for (auto idx=0; idx<refDim; ++idx) {
                for (auto jdx=0; jdx<refDim; ++jdx) {
                    // the expected value
                    pixel_t expected = pid;
                    // the actual value
                    pixel_t actual = ref[idx*refDim + jdx];
                    // compute the mismatch
                    auto mismatch = std::abs(expected-actual)/std::abs(expected);
                    // if there is a mismatch
                    if (mismatch > std::numeric_limits<value_t>::epsilon()) {
                        // make a channel
                        pyre::journal::error_t error("ampcor.cuda");
                        // complain
                        error
                            << pyre::journal::at(__HERE__)
                            << "ref[" << pid << "; " << idx << ", " << jdx << "] : mismatch: "
                            << "expected: " << expected
                            << ", actual: " << actual
                            << pyre::journal::endl;
                        // and bail
                        throw std::runtime_error("verification error");
                    }
                }
            }

            // get the secondary raster
            auto sec = arena + pid*cellsPerPair + refCells;
            // verify its contents
            for (auto idx=0; idx<refDim; ++idx) {
                for (auto jdx=0; jdx<refDim; ++jdx) {
                    // the bounds of the copy of the ref tile in the sec tile
                    auto within = (idx >= i && idx < i+refDim && jdx >= j && idx < j+refDim);
                    // the expected value depends on whether we are within the magic subtile
                    pixel_t expected = within ? ref[idx*refDim + jdx] : 0;
                    // the actual value
                    pixel_t actual = sec[idx*secDim + jdx];
                    // compute the mismatch
                    auto mismatch = std::abs(expected-actual)/std::abs(actual);
                    // if there is a mismatch
                    if (mismatch > std::numeric_limits<value_t>::epsilon()) {
                        // make a channel
                        pyre::journal::error_t error("ampcor.cuda");
                        // complain
                        error
                            << pyre::journal::at(__HERE__)
                            << "sec[" << pid << "; " << idx << ", " << jdx << "] : mismatch: "
                            << "expected: " << expected
                            << ", actual: " << actual
                            << pyre::journal::endl;
                        // and bail
                        throw std::runtime_error("verification error");
                    }
                }
            }
        }
    }
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "verifying reference dataset: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // show me
    c.dump();

    // all done
    return 0;
}

// end of file
