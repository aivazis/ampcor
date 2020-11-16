// -*- C++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// configuration
#include <portinfo>
// STL
#include <complex>
// pyre
#include <pyre/journal.h>
// cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>
// pull the declarations
#include "kernels.h"


// we use floats
using value_t = float;


// the correlation kernel
template <std::size_t T, typename valueT = value_t>
__global__
void
_correlate(const valueT * refArena, const valueT * refStats,
           const valueT * secArena, const valueT * secStats,
           std::size_t refRows, std::size_t refCols,
           std::size_t secRows, std::size_t secCols,
           std::size_t corRows, std::size_t corCols,
           std::size_t row, std::size_t col,
           valueT * correlation);


// implementation
void
ampcor::cuda::kernels::
correlate(const value_t * refArena, const value_t * refStats,
          const value_t * secArena, const value_t * secStats,
          std::size_t pairs,
          std::size_t refRows, std::size_t refCols,
          std::size_t secRows, std::size_t secCols,
          std::size_t corRows, std::size_t corCols,
          value_t * dCorrelation)
{
    // make a channel
    pyre::journal::debug_t channel("ampcor.cuda");

    // figure out the job layout and launch the calculation on the device
    // each thread block takes care of one tile pair, so we need as many blocks as there are pairs
    auto B = pairs;
    // the computation of the correlation matrix is a reduction that starts with each thread
    // handling a pair of columns, one from the reference tile and one from a particular
    // placement of the chip within the search window; this means that the number of threads
    // per block is determined by the number of columns in the reference tile
    auto T = refCols;
    // each thread stores in shared memory the partial sum for the numerator term and the
    // partial sum for the secondary tile variance; so we need two {value_t}'s worth of shared
    // memory for each thread, but no less than 64
    auto S = 2 * std::max(T, 64ul) * sizeof(value_t);

    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " blocks of " << T << " threads each, with "
        << S << " bytes of shared memory per block, for each of the "
        << "(" << corRows << "x" << corCols << ")"
        << " possible placements of the search window within the secondary tile;"
        << " a grand total of " << (B*corRows*corCols) << " kernel launches"
        << pyre::journal::endl;

    // for storing error codes
    cudaError_t status = cudaSuccess;
    // go through all possible row offsets for the sliding window
    for (auto row = 0; row < corRows; ++row) {
        // and all possible column offsets
        for (auto col = 0; col < corCols; ++col) {
            // deduce the correct kernel to launch and deploy
            // N.B.: kernel launch is an implicit barrier, so no need for any extra
            // synchronization
            if (refCols <= 32) {
                // tell me
                channel << "deploying the 32 column kernel";
                // do it
                _correlate<32> <<<B,32,S>>> (refArena, refStats, secArena, secStats,
                                             refRows, refCols, secRows, secCols, corRows, corCols,
                                             row, col, dCorrelation);
            } else if (refCols <= 64) {
                // tell me
                channel << "deploying the 64 column kernel";
                // do it
                _correlate<64> <<<B,64,S>>> (refArena, refStats, secArena, secStats,
                                             refRows, refCols, secRows, secCols, corRows, corCols,
                                             row, col, dCorrelation);
            } else if (refCols <= 128) {
                // tell me
                channel << "deploying the 128 column kernel";
                // do it
                _correlate<128> <<<B,128,S>>> (refArena, refStats, secArena, secStats,
                                               refRows, refCols, secRows, secCols, corRows, corCols,
                                               row, col, dCorrelation);
            } else if (refCols <= 256) {
                // tell me
                channel << "deploying the 256 column kernel";
                // do it
                _correlate<256> <<<B,256,S>>> (refArena, refStats, secArena, secStats,
                                               refRows, refCols, secRows, secCols, corRows, corCols,
                                               row, col, dCorrelation);
            } else if (refCols <= 512) {
                // tell me
                channel << "deploying the 512 column kernel";
                // do it
                _correlate<512> <<<B,512,S>>> (refArena, refStats, secArena, secStats,
                                               refRows, refCols, secRows, secCols, corRows, corCols,
                                               row, col, dCorrelation);
            } else {
                // complain
                throw std::runtime_error("cannot handle reference tiles of this shape");
            }
            // check for errors
            status = cudaPeekAtLastError();
            // if something went wrong
            if (status != cudaSuccess) {
                // make a channel
                pyre::journal::error_t error("ampcor.cuda");
                // complain
                error
                    << pyre::journal::at(__HERE__)
                    << "after launching the " << row << "x" << col << " correlators: "
                    << cudaGetErrorName(status) << " (" << status << ")"
                    << pyre::journal::endl;
                // and bail
                break;
            }
        }
        // if something went wrong in the inner loop
        if (status != cudaSuccess) {
            // bail out of the outer loop as well
            break;
        }
    }
    // wait for the device to finish
    status = cudaDeviceSynchronize();
    // check
    if (status != cudaSuccess) {
        // get the error description
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while waiting for a kernel to finish: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::runtime_error(description);
    }

    // all done
    return;
}


// the correlation kernel
template <std::size_t T, typename valueT>
__global__
void
_correlate(const valueT * refArena, // the reference tiles
           const valueT * refStats, // the hyper-grid of reference tile variances
           const valueT * secArena, // the secondary tiles
           const valueT * secStats, // the hyper-grid of secondary tile averages
           std::size_t refRows, std::size_t refCols,
           std::size_t secRows, std::size_t secCols,
           std::size_t corRows, std::size_t corCols,
           std::size_t row, std::size_t col,
           valueT * correlation)
{

    // build the workload descriptors
    // global
    // std::size_t B = gridDim.x;    // number of blocks
    // std::size_t T = blockDim.x;   // number of threads per block
    // std::size_t W = B*T;          // total number of workers
    // local
    std::size_t b = blockIdx.x;      // my block id
    std::size_t t = threadIdx.x;     // my thread id within my block
    // std::size_t w = b*T + t;      // my worker id


    // N.B.: do not be tempted to terminate early threads that have no assigned workload; their
    // participation is required to make sure that shared memory is properly zeroed out for the
    // nominally out of bounds accesses

    // get access to my shared memory
    extern __shared__ valueT scratch[];
    // get a handle to this thread block group
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();

    // initialize the numerator term
    valueT numerator = 0;
    // initialize the secondary variance accumulator
    valueT secVariance = 0;
    // look up the mean secondary amplitude: skip over the cells handled by other block, and
    // then skip over the cells handled by other threads in my block
    auto mean = secStats[b*corRows*corCols + row*corCols + col];

    // my {ref} starting point is column {t} of grid {b}
    auto ref = refArena + b*refRows*refCols + t;
    // my {sec} starting point is column {t} of the slice of grid {b} at (row, col)
    auto sec = secArena + b*secRows*secCols + (row*secCols + col) + t;

    // if my thread id is less than the number of columns in the reference tile, i need to sum
    // up the contributions to the numerator and the secondary tile variance from my column; if
    // not, my contribution is to zero out my slots in shared memory so the reduction doesn't
    // read uninitialized memory
    if (t < refCols) {
        // run down the two matching columns, one from {ref}, one from {sec}
        for (std::size_t idx=0; idx < refCols; ++idx) {
            // fetch the {ref} value
            valueT r = ref[idx*refCols];
            // fetch the {sec} value and subtract the mean secondary amplitude
            valueT t = sec[idx*secCols] - mean;
            // update the numerator
            numerator += r * t;
            // and the secondary variance
            secVariance += t * t;
        }
    }

    // save my partial results; idle threads only do this bit, with {numerator} and
    // {secVariance} still at their initial values
    scratch[2*t] = numerator;
    scratch[2*t + 1] = secVariance;
    // barrier: make sure everybody is done
    cta.sync();

    // now do the reduction in shared memory
    // for progressively smaller block sizes, the bottom half of the threads collect partial sums
    // N.B.: T is a template parameter, known at compile time, so it's easy for the optimizer to
    // eliminate the impossible clauses
    // for 512 threads per block
    if (T >= 512 && t < 256) {
        // my sibling's offset
        auto offset = 2*(t+256);
        // update my partial sum by reading my sibling's value
        numerator += scratch[offset];
        // ditto for the secondary variance
        secVariance += scratch[offset+1];
        // and make them available
        scratch[2*t] = numerator;
        scratch[2*t+1] = secVariance;
    }
    // make sure everybody is done
    cta.sync();

    // for 256 threads per block
    if (T >= 256 && t < 128) {
        // my sibling's offset
        auto offset = 2*(t+128);
        // update my partial sum by reading my sibling's value
        numerator += scratch[offset];
        // ditto for the secondary variance
        secVariance += scratch[offset+1];
        // and make them available
        scratch[2*t] = numerator;
        scratch[2*t+1] = secVariance;
    }
    // make sure everybody is done
    cta.sync();

    // for 128 threads per block
    if (T >= 128 && t < 64) {
        // my sibling's offset
        auto offset = 2*(t+64);
        // update my partial sum by reading my sibling's value
        numerator += scratch[offset];
        // ditto for the secondary variance
        secVariance += scratch[offset+1];
        // and make them available
        scratch[2*t] = numerator;
        scratch[2*t+1] = secVariance;
    }
    // make sure everybody is done
    cta.sync();

    // on recent architectures, there is a faster way to do the reduction once we reach the
    // warp level; the only cost is that we have to make sure there is enough memory for 64
    // threads, i.e. the shared memory size is bound from below by 64*sizeof(valueT)
    if (t < 32) {
        // if we need to
        if (T >= 64) {
            // my sibling's offset
            auto offset = 2*(t+32);
            // pull a neighbor's value
            numerator += scratch[offset];
            secVariance += scratch[offset+1];
        }
        // get a handle to the active thread group
        cooperative_groups::coalesced_group active = cooperative_groups::coalesced_threads();
        // the power-of-2 threads
        for (int offset = 16; offset > 0; offset >>= 1) {
            // reduce using {shuffle}
            numerator += active.shfl_down(numerator, offset);
            secVariance += active.shfl_down(secVariance, offset);
        }
    }

    // finally, the master thread of each block
    if (t == 0) {
        // looks up the sqrt of the reference tile variance
        valueT refVariance = refStats[b];
        // computes the correlation
        auto corr = numerator / refVariance / std::sqrt(secVariance);
        // computes the slot where this result goes
        std::size_t slot = b*corRows*corCols + row*corCols + col;
        // and writes the sum to the result vector
        correlation[slot] = corr;
    }

    // all done
    return;
}


// end of file
