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

    // each thread moves the data in its column to a new location, so we need as many threads
    // as there are columns in a narrow tile, rounded up to the nearest warp
    const auto T = 32 * (tileCols / 32 + (tileCols % 32 ? 1 : 0));
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

    // convert to {thrust::complex}
    auto tarena = reinterpret_cast<complex_t *>(arena);
    // launch
    _spread<<<B,T>>> (tarena, arenaRows, arenaCols, tileRows, tileCols);

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
            << "while launching the spread kernel: "
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
            << "while spreading the spectrum of the refined tiles: "
            << description << " (" << execStatus << ")"
            << pyre::journal::endl;
        // bail
        throw std::runtime_error(description);
    }

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

    // this is the transformation we apply to each tile in the arena
    //
    //     X X . . .          X . . . X
    //     X X . . .          . . . . .
    //     . . . . .   --->   . . . . .
    //     . . . . .          . . . . .
    //     . . . . .          X . . . X
    //
    // where the overall shape is in {arena} and the shape of the X blocks in {narrowShape}
    // each thread handles one column

    // N.B.: if {tileCols} is not divisible by 2, the X blocks are not all the same size, so we
    // have to do the integer arithmetic carefully; to keep things straight, let's number the
    // blocks:
    //
    //     0 1 . . .         0 . . . 1
    //     2 3 . . .         . . . . .
    //     . . . . .  --->   . . . . .
    //     . . . . .         . . . . .
    //     . . . . .         2 . . . 3
    //

    // block 0 is, by definition, (tileRows/2, tileCols/2), where integer division is understood
    // this fixes everybody else's size:
    // block 1: (tileRows, tileCols - tileCols/2)
    // block 2: (tileRows - tileRows/2, tileCols)
    // block 3: (tileRows - tileRows/2, tileCols - tileCols/2)

    // make a zero
    complexT zero {0, 0};
    // the beginning of my tile is at
    auto tile = arena + b * (arenaRows*arenaCols);

    // in what follows, the number of columns of the block being moved is implicit in the range
    // of the ids of the threads that handle it; the number of rows shows up explicitly as a
    // loop counter

    // move block 1
    if (t >= tileCols/2 && t < tileCols) {
        // the source column starts at
        auto src = tile + t;
        // the destination column starts at
        auto dst = src + arenaCols - (tileCols - tileCols/2);
        // run down the column
        for (auto row = 0; row < tileRows/2; ++row) {
            // copy my data to its destination
            dst[0] = src[0];
            // zero out the source cell
            src[0] = zero;
            // update the pointers to get to the next row
            src += arenaCols;
            dst += arenaCols;
        }
    }

    // move block 2
    if (t < tileCols/2) {
        // my column in the source row starts at
        auto src = tile + (tileRows/2 * arenaCols) + t;
        // the destination is a few rows below me
        auto dst = src + (arenaRows - tileRows/2)*arenaCols;
        // run down the column
        for (auto row = 0; row < tileRows - tileRows/2; ++row) {
            // copy my data to its destination
            dst[0] = src[0];
            // zero out the source cell
            src[0] = zero;
            // update the pointers to get to the next row
            src += arenaCols;
            dst += arenaCols;
        }
    }

    // move block 3
    if (t >= tileCols/2 && t < tileCols) {
        // the source column starts at
        auto src = tile + t;
        // the destination is a few rows below me and a few columns to the right
        auto dst = src + (arenaRows-tileRows/2)*arenaCols + arenaCols-(tileCols-tileCols/2);
        // run down the column
        for (auto row=0; row < tileRows - tileRows/2; ++row) {
            // copy my data to its destination'
            dst[0] = src[0];
            // zero out the source cell
            src[0] = zero;
            // update the pointers
            src += arenaCols;
            dst += arenaCols;
        }
    }

    // all done
    return;
}


// end of file
