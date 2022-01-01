// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2022 all rights reserved
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
        int pairs, int corRows, int corCols,
        int orgRow, int orgCol, value_t zoomFactor,
        value_t * loc);


// run through the correlation matrix for each, find its maximum value and record its location
void
ampcor::cuda::kernels::
maxcor(const float * gamma,
       int pairs, int corRows, int corCols,
       int orgRow, int orgCol, float zoomFactor,
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


// the kernel that locates the maximum value of the correlation surface and records the shift
template <typename value_t>
__global__
void
_maxcor(const value_t * gamma,
        int pairs, int corRows, int corCols,
        int orgRow, int orgCol, value_t zoomFactor,
        value_t * loc)
{
    // build the workload descriptors
    // global
    // auto B = gridDim.x;    // number of blocks
    auto T = blockDim.x;      // number of threads per block
    // auto W = B*T;          // total number of workers
    // local
    auto b = blockIdx.x;      // my block id
    auto t = threadIdx.x;     // my thread id within my block
    auto w = b*T + t;         // my worker id

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
