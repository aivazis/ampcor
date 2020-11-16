// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//

// configuration
#include <portinfo>
// externals
#include <random>
// ampcor
#include <ampcor/correlators.h>
#include <ampcor_cuda/correlators.h>


// type aliases
using value_t = float;
// on the host
using arena_type = ampcor::dom::arena_raster_t<value_t>;
using arena_layout_type = arena_type::layout_type;
using arena_shape_type = arena_type::shape_type;
using arena_index_type = arena_type::index_type;
// on the device
using dev_arena_type = ampcor::cuda::correlators::devarena_raster_t<value_t>;


// test driver
int test() {

    // pick the number of tiles
    auto tiles = 1;
    // the shape of the reference chips
    auto refRows = 256;
    auto refCols = 264;
    // the padding
    auto padRows = 2;
    auto padCols = 4;
    // the shape of the secondary windows
    auto secRows = refRows + 2*padRows;
    auto secCols = refCols + 2*padCols;
    // and the placements
    auto corRows = 2 * padRows + 1;
    auto corCols = 2 * padCols + 1;

    // make a shape
    arena_shape_type arenaShape { tiles, secRows, secCols };
    // and an origin
    arena_index_type arenaOrigin { 0, -padRows, -padCols };
    // and a layout
    arena_layout_type arenaLayout { arenaShape, arenaOrigin };
    // build a tile arena
    arena_type hostArena { arenaLayout, "mean_host_arena.dat", arenaLayout.cells() };

    // make a generator
    std::mt19937 gen;
    // and a distribution
    std::uniform_real_distribution<value_t> uni(0,1);
    // fill the secondary arena
    for (auto & v : hostArena) {
        // with uniformly distributed numbers in [0,1)
        v = uni(gen);
    }

    // make an arena on the device
    dev_arena_type devArena { arenaLayout, arenaLayout.cells() };
    // fill it with the contents of the host arena
    auto pushStatus = cudaMemcpy(devArena.data()->data(),
                                 hostArena.data()->data(),
                                 arenaLayout.cells() * sizeof(value_t), cudaMemcpyHostToDevice);
    // if something went wrong
    if (pushStatus != cudaSuccess) {
        // build the error description
        std::string description = cudaGetErrorName(pushStatus);
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while transferring tiles to the device: "
            << description << " (" << pushStatus << ")"
            << pyre::journal::endl;
        // and bail
        throw std::logic_error(description);
    }

    // helpers
    arena_index_type drow { 0, 1, 0 };
    arena_index_type dcol { 0, 0, 1 };
    // derive the shape of the SAT
    auto satShape = arenaShape + drow + dcol;
    // move the origin to make room for the border
    auto satOrigin = arenaOrigin - drow - dcol;
    // assemble the layout
    arena_layout_type satLayout { satShape, satOrigin };

    // make a SAT on the device
    dev_arena_type dsat { satLayout, satLayout.cells() };
    // fill it
    ampcor::cuda::kernels::sat(devArena.data()->data(),
                               tiles, secRows, secCols,
                               dsat.data()->data());

    // the shape of the arena of means
    arena_shape_type meanShape { tiles, corRows, corCols };
    // its origin
    arena_index_type meanOrigin { 0, -padRows, -padCols };
    // assemble its layout
    arena_layout_type meanLayout { meanShape, meanOrigin };
    // make an arena of mean values on the device
    dev_arena_type dmean { meanLayout, meanLayout.cells() };
    // fill it
    ampcor::cuda::kernels::secStats(dsat.data()->data(),
                                    tiles,
                                    refRows, refCols, secRows, secCols, corRows, corCols,
                                    dmean.data()->data());

    // make a channel
    pyre::journal::debug_t cmp("ampcor.cmp");
    // go through the table
    for (auto idx : dmean.layout()) {
        // and dump its values
        cmp
            << "dmean[" << idx << "] = " << dmean[idx]
            << pyre::journal::newline;
    }
    // flush
    cmp << pyre::journal::endl(__HERE__);

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
