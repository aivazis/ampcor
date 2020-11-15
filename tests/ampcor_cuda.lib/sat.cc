// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//

// configuration
#include <portinfo>
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
    // make a shape
    arena_shape_type arenaShape { 1024, 193, 65 };
    // and an origin
    arena_index_type arenaOrigin { 0, 0, 0 };
    // and a layout
    arena_layout_type arenaLayout { arenaShape, arenaOrigin };
    // build a tile arena
    arena_type hostArena { arenaLayout, "sat_host_arena.dat", arenaLayout.cells() };
    // fill it
    for (auto & v : hostArena) {
        // with ones
        v = 1;
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
    // and allocate the host SAT
    arena_type hostSAT { satLayout, "sat_host_sat.dat", satLayout.cells() };
    // fill it
    for (auto idx : arenaLayout) {
        // with the running sum of the tile entries
        hostSAT[idx] =
            hostArena[idx] +
            hostSAT[idx-drow] + hostSAT[idx-dcol] - hostSAT[idx-drow-dcol];
    }

    // show me
    pyre::journal::debug_t channel("ampcor.sat");
    // by visiting the table
    for (auto idx : satLayout) {
        // and dumping the values
        channel
            << "sat[" << idx << "] = " << hostSAT[idx]
            << pyre::journal::newline;
    }
    // flush
    channel << pyre::journal::endl;

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

    // make a SAT on the device
    dev_arena_type dsat { satLayout, satLayout.cells() };
    // unpack the arena shape
    auto [tiles, arenaRows, arenaCols] = arenaShape;
    // fill it
    ampcor::cuda::kernels::sat(devArena.data()->data(),
                               tiles, arenaRows, arenaCols,
                               dsat.data()->data());

    // allocate an arena on the host so we can harvest the SAT
    arena_type devSAT { satLayout, "sat_dev_sat.dat", satLayout.cells() };
    // harvest
    auto pullStatus = cudaMemcpy(devSAT.data()->data(),
                                 dsat.data()->data(),
                                 satLayout.cells() * sizeof(value_t), cudaMemcpyDeviceToHost);
    // if something went wrong
    if (pullStatus != cudaSuccess) {
        // build the error description
        std::string description = cudaGetErrorName(pullStatus);
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while transferring tiles to the device: "
            << description << " (" << pullStatus << ")"
            << pyre::journal::endl;
        // and bail
        throw std::logic_error(description);
    }

    // make a channel
    pyre::journal::info_t cmp("ampcor.cmp");
    // with a complete set of indices of the two SATs
    for (auto idx : satLayout) {
        // compare
        if (hostSAT[idx] != devSAT[idx]) {
            cmp
                << "mismatch at idx=[" << idx << "]:" << pyre::journal::newline
                << "  host=" << hostSAT[idx] << " != " << " device=" << devSAT[idx]
                << pyre::journal::newline;
        }
    }
    // flush
    cmp << pyre::journal::endl;

    // all done
    return 0;
}


// entry point
int main()
{
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
