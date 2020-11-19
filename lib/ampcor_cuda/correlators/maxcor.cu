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
_maxcor(const value_t * gamma,
        std::size_t pairs, std::size_t corRows, std::size_t corCols,
        std::size_t orgRow, std::size_t orgCol, value_t zoomFactor,
        value_t * loc);


// run through the correlation matrix for each, find its maximum value and record its location
void
ampcor::cuda::kernels::
maxcor(const float * gamma,
       std::size_t pairs, std::size_t corRows, std::size_t corCols,
       std::size_t orgRow, std::size_t orgCol, float zoomFactor,
       float * loc)
{
    // make a channel
    pyre::journal::debug_t channel("ampcor.cuda");

    // launch blocks of T threads
    auto T = 128;
    // in as many blocks as it takes to handle all pairs
    auto B = pairs / T + (pairs % T ? 1 : 0);
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " blocks of " << T
        << " threads each to handle the " << pairs
        << " entries of the correlation hyper-matrix"
        << pyre::journal::endl;
    // launch the kernels
    _maxcor <<<B,T>>> (gamma, pairs, corRows, corCols, orgRow, orgCol, zoomFactor, loc);
    // wait for the kernels to finish
    cudaError_t status = cudaDeviceSynchronize();
    // check
    if (status != cudaSuccess) {
        // get the description of the error
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while computing the average amplitudes of all possible search window placements: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::runtime_error(description);
    }

    // all done
    return;
}


// the SAT generation kernel
template <typename value_t>
__global__
void
_maxcor(const value_t * gamma,
        std::size_t pairs, std::size_t corRows, std::size_t corCols,
        std::size_t orgRow, std::size_t orgCol, value_t zoomFactor,
        value_t * loc)
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
    if (w >= pairs) {
        // nothing for me to do
        return;
    }

    // locate the beginning of my correlation matrix
    auto cor = gamma + w*corRows*corCols;
    // locate the beginning of my stats table
    auto myloc = loc + 2*w;

    // initialize
    // the maximum value
    auto high = cor[0];
    // and its location
    myloc[0] = 0;
    myloc[1] = 0;

    // go through all the cells in my matrix
    for (auto row = 0; row < corRows; ++row) {
        for (auto col = 0; col < corCols; ++col) {
            // get the value
            auto value = cor[row*corCols + col];
            // if it is higher than the current max
            if (value > high) {
                // update the current max
                high = value;
                // and its location
                myloc[0] = row;
                myloc[1] = col;
            }
        }
    }

    // shift back to the origin and apply the zoom factor
    myloc[0] = (myloc[0] + orgRow) / zoomFactor;
    myloc[1] = (myloc[1] + orgRow) / zoomFactor;

    // all done
    return;
}


// end of file
