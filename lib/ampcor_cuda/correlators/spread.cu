// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// configuration
#include <portinfo>
// STL
#include <exception>
#include <string>
// cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/complex.h>
// pyre
#include <pyre/journal.h>
// local declarations
#include "kernels.h"

// type alias
using complex_t = thrust::complex<float>;

// helpers
template <typename complexT = complex_t>
__global__
static void
_spread(complexT * arena,
        std::size_t arenaRows, std::size_t arenaCols, std::size_t tileRows, std::size_t tileCols);


// spread the tiles in the arena
void
ampcor::cuda::kernels::
spread(std::complex<float> * arena,
       std::size_t pairs, std::size_t arenaRows, std::size_t arenaCols,
       std::size_t tileRows, std::size_t tileCols)
{
    // make a channel
    pyre::journal::info_t channel("ampcor.cuda.spread");

    // each thread handles a row and a column to get the horizontal and vertical accumulated
    // phase, then we do a reduction; so we need enough workers to cover the largest tile rank
    const auto w = std::max(tileRows, tileCols);
    // round up to the nearest warp
    const auto T = 32 * (w / 32 + (w % 32 ? 1 : 0));
    // the number of blocks
    const auto B = pairs;

    // it's not 1980 any more
    auto pl = pairs == 1 ? "" : "s";
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " block" << pl <<" of " << T
        << " threads each to spread " << pairs << " tile" << pl <<" of ("
        << tileRows << "x" << tileCols << ") pixels"
        << pyre::journal::endl;

    // launch
    _spread<<<B,T>>> (arena, arenaRows, arenaCols, tileRows, tileCols);

    // all done
    return;
}


// implementation
template <typename complexT>
__global__
static void
_spread(complexT * arena,
        std::size_t arenaRows, std::size_t arenaCols, std::size_t tileRows, std::size_t tileCols)
{
    // build the workload descriptors
    // global
    // auto B = gridDim.x;      // number of blocks
    // auto T = blockDim.x;     // number of threads per block; here, a template parameter
    // auto W = B*T;            // total number of workers
    // local
    auto b = blockIdx.x;        // my block id
    auto t = threadIdx.x;       // my thread id
    // auto w = b*T + t;        // my worker id

    // all done
    return;
}


// end of file
