// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// configuration
#include <portinfo>
// cuda
#include <cuda_runtime.h>
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
        << "test: adding reference and target tile pairs to the correlator"
        << pyre::journal::endl;

    // the reference tile extent
    int refExt = 128;
    // the margin around the reference tile
    int margin = 32;
    // therefore, the target tile extent
    int tgtExt = refExt + 2*margin;
    // the number of possible placements of the reference tile within the target tile
    int placements = 2*margin + 1;
    // the number of pairs
    slc_t::size_type pairs = placements*placements;

    // the number of cells in a reference tile
    slc_t::size_type refCells = refExt * refExt;
    // the number of cells in a target tile
    slc_t::size_type tgtCells = tgtExt * tgtExt;
    // the number of cells per pair
    slc_t::size_type cellsPerPair = refCells + tgtCells;

    // the reference shape
    slc_t::shape_type refShape = {refExt, refExt};
    // the search window shape
    slc_t::shape_type tgtShape = {tgtExt, tgtExt};

    // the reference layout with the given shape and default packing
    slc_t::layout_type refLayout = { refShape };
    // the search window layout with the given shape and default packing
    slc_t::layout_type tgtLayout = { tgtShape };

    // start the clock
    timer.reset().start();
    // make a correlator
    correlator_t c(pairs, refLayout, tgtLayout);
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

            // make a target tile
            slc_t tgt(tgtLayout);
            // fill it with zeroes
            std::fill(tgt.view().begin(), tgt.view().end(), 0);
            // make a slice
            slc_t::slice_type slice = tgt.layout().slice({i,j}, {i+refExt, j+refExt});
            // make a view of the tgt tile over this slice
            slc_t::view_type view = tgt.view(slice);
            // fill it with ones
            std::copy(ref.view().begin(), ref.view().end(), view.begin());

            // show me
            channel << "tgt[" << i << "," << j << "]:" << pyre::journal::newline;
            for (auto idx=0; idx<tgtExt; ++idx) {
                for (auto jdx=0; jdx<tgtExt; ++jdx) {
                    channel << tgt[{idx, jdx}] << " ";
                }
                channel << pyre::journal::newline;
            }
            channel << pyre::journal::endl;

            // add this pair to the correlator
            c.addReferenceTile(pid, ref.constview());
            c.addTargetTile(pid, tgt.constview());
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
            for (auto idx=0; idx<refExt; ++idx) {
                for (auto jdx=0; jdx<refExt; ++jdx) {
                    // the expected value
                    pixel_t expected = pid;
                    // the actual value
                    pixel_t actual = ref[idx*refExt + jdx];
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

            // get the target raster
            auto tgt = arena + pid*cellsPerPair + refCells;
            // verify its contents
            for (auto idx=0; idx<refExt; ++idx) {
                for (auto jdx=0; jdx<refExt; ++jdx) {
                    // the bounds of the copy of the ref tile in the tgt tile
                    auto within = (idx >= i && idx < i+refExt && jdx >= j && idx < j+refExt);
                    // the expected value depends on whether we are within the magic subtile
                    pixel_t expected = within ? ref[idx*refExt + jdx] : 0;
                    // the actual value
                    pixel_t actual = tgt[idx*tgtExt + jdx];
                    // compute the mismatch
                    auto mismatch = std::abs(expected-actual)/std::abs(actual);
                    // if there is a mismatch
                    if (mismatch > std::numeric_limits<value_t>::epsilon()) {
                        // make a channel
                        pyre::journal::error_t error("ampcor.cuda");
                        // complain
                        error
                            << pyre::journal::at(__HERE__)
                            << "tgt[" << pid << "; " << idx << ", " << jdx << "] : mismatch: "
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
