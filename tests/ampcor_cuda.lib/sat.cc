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


// driver
int main() {
    // make a shape
    arena_shape_type arenaShape { 1, 32, 32 };
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
        hostSAT[idx] =
            hostArena[idx] +
            hostSAT[idx-drow] + hostSAT[idx-dcol] - hostSAT[idx-drow-dcol];
    }

    // show me
    pyre::journal::debug_t channel("ampcor.sat");
    // by visiting the table
    for (auto idx : satLayout) {
        channel
            << "sat[" << idx << "] = " << hostSAT[idx]
            << pyre::journal::newline;
    }
    // flush
    channel << pyre::journal::endl;

    // make an arena on the device
    dev_arena_type devArena { arenaLayout, arenaLayout.cells() };
    // fill it with the contents of the host arena
    cudaMemcpy(devArena.data()->data(),
               hostArena.data()->data(),
               arenaLayout.cells() * sizeof(value_t), cudaMemcpyHostToDevice);

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
    cudaMemcpy(devSAT.data()->data(),
               dsat.data()->data(),
               satLayout.cells() * sizeof(value_t), cudaMemcpyDeviceToHost);

    // make a channel
    pyre::journal::info_t cmp("ampcor.cmp");
    // with a complete set of indices of the two SATs
    for (auto idx : satLayout) {
        // compare
        if (hostSAT[idx] != devSAT[idx]) {
            cmp
                << "mismatch at idx=[" << idx << "]:" << pyre::journal::newline
                << "  host <- " << hostSAT[idx] << " != " << " device <- " << devSAT[idx]
                << pyre::journal::newline;
        }
    }
    // flush
    cmp << pyre::journal::endl;

    // all done
    return 0;
}

// end of file
