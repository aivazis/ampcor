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
// pyre
#include <pyre/journal.h>
// local declarations
#include "kernels.h"


// helpers
__global__
static void
_deramp(cuFloatComplex * arena,
        std::size_t arenaRows, std::size_t arenaCols, std::size_t tileRows, std::size_t tileCols);


// deramp the tiles in the arena
void
ampcor::cuda::kernels::
deramp(std::complex<float> * arena,
       std::size_t pairs, std::size_t arenaRows, std::size_t arenaCols,
       std::size_t tileRows, std::size_t tileCols)
{
    // make a channel
    pyre::journal::info_t channel("ampcor.cuda.deramp");

    // we need enough workers to cover the largest tile rank
    auto workers = std::max(tileRows, tileCols);
    // round the number of threads up to the nearest full warp
    auto T = 32 * (workers / 32 + (workers % 32 ? 1 : 0));
    // the number of block
    auto B = pairs;
    // the amount of shared memory to set aside; we do warp shuffling, so there is a low bound
    auto S = std::max(64ul, T) * sizeof(float);

    // it's not 1980 any more
    auto pl = pairs == 1 ? "" : "s";
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " block" << pl <<" of " << T
        << " threads each to deramp " << pairs << " tile" << pl <<" of ("
        << tileRows << "x" << tileCols << ") pixels"
        << pyre::journal::endl;

    // REDUCTION TEMPLATE EXPANSION BASED ON T
    // launch the SAT kernel
    _deramp <<<B,T,S>>> (reinterpret_cast<cuFloatComplex *>(arena),
                         arenaRows, arenaCols, tileRows, tileCols);


    // check whether all went well
    auto launchStatus = cudaGetLastError();
    // if something went wrong
    if (launchStatus != cudaSuccess) {
        // form the error description
        std::string description = cudaGetErrorName(launchStatus);
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while launching the deramp kernel: "
            << description << " (" << launchStatus << ")"
            << pyre::journal::endl;
        // bail
        throw std::runtime_error(description);
    }
    // wait for the device to finish
    auto execStatus = cudaDeviceSynchronize();
    // if something went wrong
    if (execStatus != cudaSuccess) {
        // form the error description
        std::string description = cudaGetErrorName(execStatus);
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while computing the deramp phase: "
            << description << " (" << execStatus << ")"
            << pyre::journal::endl;
        // bail
        throw std::runtime_error(description);
    }

    // all done
    return;
}


// implementation
__global__
static void
_deramp(cuFloatComplex * arena,
        std::size_t arenaRows, std::size_t arenaCols, std::size_t tileRows, std::size_t tileCols)
{
    // build the workload descriptors
    // global
    // std::size_t B = gridDim.x;      // number of blocks
    // std::size_t T = blockDim.x;        // number of threads per block
    // auto W = B*T;                   // total number of workers
    // local
    std::size_t b = blockIdx.x;        // my block id
    std::size_t t = threadIdx.x;       // my thread id
    // auto w = b*T + t;               // my worker id

    // get access to my shared memory
    extern __shared__ float scratch[];

    // find the beginning of my tile by skipping the ones done by other blocks
    auto tile = arena + b * (arenaRows*arenaCols);

    // all done
    return;
}


// end of file
