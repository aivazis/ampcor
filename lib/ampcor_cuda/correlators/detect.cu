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
#include <thrust/complex.h>
// pyre
#include <pyre/journal.h>
// local declarations
#include "kernels.h"


// type aliases
using real_t = float;
using complex_t = thrust::complex<real_t>;

// helpers
template <typename complexT = complex_t>
__global__
static void
_detect(const complexT * cArena, std::size_t cells, std::size_t load,
        typename complexT::value_type * rArena);


// compute the amplitude of the signal tiles, assuming pixels are of type {complex_t}
void
ampcor::cuda::kernels::
detect(const std::complex<float> * cArena,
       std::size_t pairs, std::size_t rows, std::size_t cols,
       float * rArena)
{
    // the total number of cells
    auto cells = pairs * rows * cols;

    // this is embarrassingly parallel, so pick a simple deployment schedule
    // the load of each thread
    std::size_t N = 128*128;
    // the number of threads per block
    std::size_t T = 128;
    // the number of cells handled by a block
    std::size_t L = N*T;
    // hence, the number of blocks
    std::size_t B = (cells / L) + (cells % L ? 1 : 0);

    // make a channel
    pyre::journal::debug_t channel("ampcor.cuda");
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " blocks of " << T << " threads each to process "
        << cells << " cells"
        << pyre::journal::endl;

    // launch
    _detect <<<B,T>>> (reinterpret_cast<const complex_t *>(cArena), cells, N, rArena);
    // wait for the device to finish
    cudaError_t status = cudaDeviceSynchronize();
    // if something went wrong
    if (status != cudaSuccess) {
        // form the error description
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while computing pixel amplitudes: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // bail
        throw std::runtime_error(description);
    }

    // all done
    return;
}


// implementations
template <typename complexT>
__global__
static void
_detect(const complexT * cArena, std::size_t cells, std::size_t load,
        typename complexT::value_type * rArena)
{
    // build the workload descriptors
    // global
    // auto B = gridDim.x;             // number of blocks
    auto T = blockDim.x;               // number of threads per block
    // auto W = B*T;                   // total number of workers
    // local
    auto b = blockIdx.x;               // my block id
    auto t = threadIdx.x;              // my thread id
    // auto w = b*T + t;               // my worker id

    // the number of cells handled by each block
    auto L = T * load;
    // the number of cells handled by the blocks before me
    auto begin = b*L;
    // threads in this block should go no further than
    auto end = min((b+1)*L, cells);

#if defined(DEBUG_DETECT)
    // the first thread of each block
    if (t == 0) {
        // show me
        printf("[%05lu]: begin=%lu, end=%lu\n", w, begin, end);
    }
#endif

    // go through my cells
    for (auto current=begin+t; current < end; current += T) {
        // get the complex pixel, compute its amplitude and store it
        rArena[current] = thrust::abs(cArena[current]);
    }

    // all done
    return;
}


// end of file
