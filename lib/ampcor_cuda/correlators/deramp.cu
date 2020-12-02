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
template <std::size_t T, typename complexT = complex_t>
__global__
static void
_deramp(complexT * arena,
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

    // each thread handles a row and a column to get the horizontal and vertical accumulated
    // phase, then we do a reduction; so we need enough workers to cover the largest tile rank
    const auto w = std::max(tileRows, tileCols);
    // the number of blocks
    const auto B = pairs;
    // each worker stores two complex numbers: the horizontal and vertical phase accumulator
    const auto s = 2 * sizeof(complex_t);

    // it's not 1980 any more
    auto pl = pairs == 1 ? "" : "s";
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " block" << pl <<" of no less than " << w
        << " threads each to deramp " << pairs << " tile" << pl <<" of ("
        << tileRows << "x" << tileCols << ") pixels"
        << pyre::journal::endl;

    // convert the arena
    auto carena = reinterpret_cast<thrust::complex<float> *>(arena);

    // deploy with warp count known at compile time
    if (w <= 32) {
        // set the number of threads
        const int T = 32;
        // the amount of shared memory to set aside; we do warp shuffling, so there is a low
        // bound of 64 slots
        const int S = 64 * s;
        // tell me
        channel << "deploying " << T << " workers";
        // launch
        _deramp<T> <<<B,T,S>>> (carena, arenaRows, arenaCols, tileRows, tileCols);
    } else if (w <= 64) {
        // set the number of threads
        const int T = 64;
        // the amount of shared memory to set aside
        const int S = T * s;
        // tell me
        channel << "deploying " << T << " workers";
        // launch
        _deramp<T> <<<B,T,S>>> (carena, arenaRows, arenaCols, tileRows, tileCols);
    } else if (w <= 128) {
        // set the number of threads
        const int T = 128;
        // the amount of shared memory to set aside; we do warp shuffling, so there is a low bound;
        const int S = T * s;
        // tell me
        channel << "deploying " << T << " workers";
        // launch
        _deramp<T> <<<B,T,S>>> (carena, arenaRows, arenaCols, tileRows, tileCols);
    } else if (w <= 256) {
        // set the number of threads
        const int T = 256;
        // the amount of shared memory to set aside; we do warp shuffling, so there is a low bound;
        const int S = T * s;
        // tell me
        channel << "deploying " << T << " workers";
        // launch
        _deramp<T> <<<B,T,S>>> (carena, arenaRows, arenaCols, tileRows, tileCols);
    } else if (w <= 512) {
        // set the number of threads
        const int T = 512;
        // the amount of shared memory to set aside; we do warp shuffling, so there is a low bound;
        const int S = T * s;
        // tell me
        channel << "deploying " << T << " workers";
        // launch
        _deramp<T> <<<B,T,S>>> (carena, arenaRows, arenaCols, tileRows, tileCols);
    } else {
        // complain
        throw std::runtime_error("while in {deramp}: cannot handle tiles of this shape");
    }
    // flush
    channel << pyre::journal::endl;

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
template <std::size_t T, typename complexT>
__global__
static void
_deramp(complexT * arena,
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

    // a small number
    const typename complexT::value_type eps = 1e-6;
    // get access to my shared memory
    extern __shared__ complexT scratch[];
    // get a handle to this thread block group
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();

    // find the beginning of my tile by skipping the ones done by other blocks
    auto tile = arena + b * (arenaRows*arenaCols);

    // initialize the phase accumulator along rows
    complexT phaseHorz { 0, 0 };
    // and along columns
    complexT phaseVert { 0, 0 };

    // if i have been assigned a row
    if (t < tileRows) {
        // first element of my row
        auto row = tile + t*arenaCols;
        // run across the row; don't forget to exclude the last element in this row since we
        // look to the right as we accumulate
        for (auto idx = 0; idx < tileCols-1; ++idx) {
            // collect
            phaseHorz += row[idx] * thrust::conj(row[idx+1]);
        }
    }

    // if I have been assigned a column
    if (t < tileCols) {
        // first element of my column
        auto col = tile + t;
        // run down my column; don't forget to exclude the last element in this column since we
        // look down as we accumulate
        for (auto idx = 0; idx < tileRows - 1; ++idx) {
            // collect
            phaseVert += col[idx] * thrust::conj(col[idx+arenaCols]);
        }
    }

    // now, we all save our partial results to shared memory; this includes the guys that have
    // been idling during either or both of the traversals; each thread is assigned two slots:
    // one for the horizontal phase accumulator and one for the vertical
    // my slot is at
    auto slot = scratch + 2*t;
    // write the horizontal accumulator
    slot[0] = phaseHorz;
    // and the vertical accumulator
    slot[1] = phaseVert;
    // barrier
    cta.sync();

    // reduction time; the following section is a loop unrolled by hand, with compile-time
    // checks to remove the unused blocks; the general strategy is to have the low half of the
    // remaining block read the accumulators of the upper, update its own, and store in shared
    // memory; we then half the block size and try again, until we hit warp size, then we
    // shuffle

#ifdef MGA_DERAMP_REDUCE_LAMBDA
    // here is a lambda version of the reducer; debug...
    auto reduce = [&] (int halfBlock) {
        // my sibling from the upper half of the block stored its accumulators here
        auto sibling = scratch + 2*(t+halfBlock);
        // update my accumulators
        phaseHorz += sibling[0];
        phaseVert += sibling[1];
        // record my values
        slot[0] = phaseHorz;
        slot[1] = phaseVert;
        // all done
        return;
    };
#endif

    // for 512 threads per block
    if (T >= 512 && t < 256) {
        // my sibling from the upper half of the block stored its accumulators here
        auto sibling = scratch + 2*(t+256);
        // update my accumulators
        phaseHorz += sibling[0];
        phaseVert += sibling[1];
        // record my values
        slot[0] = phaseHorz;
        slot[1] = phaseVert;
    }
    // barrier
    cta.sync();

    // for 256 threads per block
    if (T >= 256 && t < 128) {
        // my sibling from the upper half of the block stored its accumulators here
        auto sibling = scratch + 2*(t+128);
        // update my accumulators
        phaseHorz += sibling[0];
        phaseVert += sibling[1];
        // record my values
        slot[0] = phaseHorz;
        slot[1] = phaseVert;
    }
    // barrier
    cta.sync();

    // for 128 threads per block
    if (T >= 128 && t < 64) {
        // my sibling from the upper half of the block stored its accumulators here
        auto sibling = scratch + 2*(t+64);
        // update my accumulators
        phaseHorz += sibling[0];
        phaseVert += sibling[1];
        // record my values
        slot[0] = phaseHorz;
        slot[1] = phaseVert;
    }
    // barrier
    cta.sync();

    // ok; at this point, the half block is a warp and we can shuffle
    if (t < 32) {
        // if we need to pull values from the upper half of a block
        if (T >= 64) {
            // my sibling from the upper half of the block stored its accumulators here
            auto sibling = scratch + 2*(t+64);
            // update my accumulators, but no need to bother updating shared memory; we are
            // done with it
            phaseHorz += sibling[0];
            phaseVert += sibling[1];
        }
        // get a handle to the active thread group
        cooperative_groups::coalesced_group active = cooperative_groups::coalesced_threads();
        // the threads with power-of-2 ids
        for (auto offset = 16; offset > 0; offset >>= 1) {
            // reduce using shuffle
            phaseHorz += active.shfl_down(phaseHorz, offset);
            phaseVert += active.shfl_down(phaseVert, offset);
        }
    }
    // barrier
    cta.sync();

    // the reduced values are available only on thread 0
    if (t == 0) {
        // let's write them in a known location in shared memory so everybody has access
        scratch[0] =  phaseHorz;
        scratch[1] =  phaseVert;
    }
    // barrier
    cta.sync();

    // finally, we have to deramp the tile using the phase accumulators; we will go down
    // columns of the tile handled by this block; if i have been assigned one
    if (t < tileCols) {
        // get the complex accumulators from shared memory
        auto phaseHorz = scratch[0];
        auto phaseVert = scratch[1];

        // convert into angles
        auto phiV =
            thrust::abs(phaseVert) > eps ?
            std::atan2(phaseVert.imag(), phaseVert.real()) : 0;
        auto phiH =
            thrust::abs(phaseHorz) > eps ?
            std::atan2(phaseHorz.imag(), phaseHorz.real()) : 0;

        // find the start of my column; {t} doubles as the column index in what follows
        auto col = tile + t;
        // run down the column
        for (auto r = 0; r < tileRows; ++r) {
            // form an linear combination of the two phases that depends on the tile element
            // recall that {t} doubles as the column index
            auto comb = r * phiV + t * phiH;
            // for the phase factor
            complexT phase { std::cos(comb), std::sin(comb) };
            // multiply the value in place with this phase
            col[r*arenaRows] *= phase;
        }
    }

    // all done
    return;
}


// end of file
