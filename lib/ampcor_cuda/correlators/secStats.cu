// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2023 all rights reserved
//


// configuration
#include <portinfo>
// STL
#include <complex>
// cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>
// pyre
#include <pyre/journal.h>
// pull the declarations
#include "kernels.h"


// the SAT generation kernel
template <typename value_t = float>
__global__
static void
_secStats(const value_t * sat,
          std::size_t pairs,
          std::size_t refRows, std::size_t refCols,
          std::size_t secRows, std::size_t secCols,
          std::size_t corRows, std::size_t corCols,
          value_t * stats);


// implementation

// precompute the amplitude averages for all possible placements of a reference chip within the
// secondary search window for all pairs in the plan. we allocate room for
// {_pairs}*{corRows*corCols} floating point values and use the precomputed SAT tables resident
// on the device.
//
// the SAT tables require a slice and produce the sum of the values of cells within the slice
// in no more than four memory accesses per search tile; there are boundary cases to consider
// that add a bit of complexity to the implementation; the boundary cases could have been
// trivialized using ghost cells around the search window boundary, but the memory cost is high
void
ampcor::cuda::kernels::
secStats(const float * dSAT,
         std::size_t pairs,
         std::size_t refRows, std::size_t refCols,
         std::size_t secRows, std::size_t secCols,
         std::size_t corRows, std::size_t corCols,
         float * dStats)
{
    // make a channel
    pyre::journal::info_t channel("ampcor.cuda.secStats");

    // launch blocks of T threads
    auto T = 128;
    // in as many blocks as it takes to handle all pairs
    auto B = pairs / T + (pairs % T ? 1 : 0);
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " blocks of " << T
        << " threads each to handle the " << pairs
        << " entries of the secondary amplitude averages arena"
        << pyre::journal::endl;
    // launch the kernels
    _secStats <<<B,T>>> (dSAT,
                         pairs,
                         refRows, refCols, secRows, secCols, corRows, corCols,
                         dStats);
    // check whether all went well
    auto launchStatus = cudaGetLastError();
    // if something went wrong
    if (launchStatus != cudaSuccess) {
        // form the error description
        std::string description = cudaGetErrorName(launchStatus);
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda.secStats");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while computing the arena of placement averages"
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
            << "while computing the average amplitudes of all possible search window placements: "
            << description << " (" << execStatus << ")"
            << pyre::journal::endl;
        // bail
        throw std::runtime_error(description);
    }

    // all done
    return;
}


// the SAT generation kernel
template <typename value_t>
__global__
void
_secStats(const value_t * dSAT,
          std::size_t tiles,
          std::size_t refRows, std::size_t refCols,
          std::size_t secRows, std::size_t secCols,
          std::size_t corRows, std::size_t corCols,
          value_t * dStats)
{
    // build the workload descriptors
    // global
    // std::size_t B = gridDim.x;    // number of blocks
    std::size_t T = blockDim.x;      // number of threads per block
    // std::size_t W = B*T;          // total number of workers
    // local
    std::size_t b = blockIdx.x;      // my block id
    std::size_t t = threadIdx.x;     // my thread id within my block
    std::size_t w = b*T + t;         // my worker id

    // if my worker id exceeds the number of cells that require update
    if (w >= tiles) {
        // nothing for me to do
        return;
    }

    // the shape of the SAT table includes the ghost cells
    auto satRows = secRows + 1;
    auto satCols = secCols + 1;

    // compute the number of cells in a reference tile; it scales the running sum
    auto refCells = refRows * refCols;
    // the number of cells in a SAT; lets me skip to my SAT
    auto satCells = satRows * satCols;
    // and the number of cells in a {stats} matrix; lets me skip to my {stats} slice
    auto corCells = corRows * corCols;

    // locate the beginning of my SAT table
    auto sat = dSAT + w*satCells;
    // locate the beginning of my {stats} table
    auto stats = dStats + w*corCells;

    // fill each slot in the output table by looping over (row,col) indices
    // the {row} range
    for (auto row = 0; row < corRows; ++row) {
        // the {col} range}
        for (auto col = 0; col < corCols; ++col) {
            // computing the sum of the secondary amplitudes for this placement involves
            // reading four values from the SAT whose locations are derived from {row,col}

            // N.B.: the SAT has a border with zeroes that guard against out of bounds
            //       accesses, but we must still get the arithmetic right
            // the upper left corner
            auto iUL = row*satCols + col;
            // the upper right corner is {refCols+1} away from that
            auto iUR = iUL + refCols;
            // the lower left corner: skip (refRows+1) rows of the SAT
            auto iLL = iUL + refRows*satCols;
            // the lower right corner is just {refCols+1} away from that
            auto iLR = iLL + refCols;

            // the sum is
            auto sum = sat[iLR] - sat[iLL] - sat[iUR] + sat[iUL];

            // identify the slot we write to
            auto slot = stats + row*corCols + col;
            // store the result: the running sum scaled by the size of a reference tile
            *slot = sum / refCells;
        }
    }

    // all done
    return;
}


// end of file
