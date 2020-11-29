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
// local declarations
#include "kernels.h"


// the SAT generation kernel
template <typename value_t = float>
__global__
static void
_sat(const value_t * dArena, std::size_t tileRows, std::size_t tileCols, value_t * dSAT);


// implementation
void
ampcor::cuda::kernels::
sat(const float * dArena,
    std::size_t pairs, std::size_t tileRows, std::size_t tileCols,
    float * dSAT)
{
    // make a channel
    pyre::journal::info_t channel("ampcor.cuda.sat");

    // to compute a SAT for each tile, we launch as many thread blocks as there are tiles
    std::size_t B = pairs;
    // the number of threads per block is determined by the shape of the tiles; each
    // thread will handle one row and/or one column in the SAT table, so that means we need as
    // many threads as {max(tileRows, tileCols)} to guarantee we have enough workers; the border
    // around the SAT will be handled as an explicit pass
    auto workers = std::max(tileRows, tileCols);
    // then we round up to the nearest warp...
    std::size_t T = 32 * (workers / 32 + (workers % 32 ? 1 : 0));
    // it's not 1980 any more
    auto pl = pairs == 1 ? "" : "s";
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " block" << pl <<" of " << T
        << " threads each to compute SATs for " << pairs << " tile" << pl <<" of ("
        << tileRows << "x" << tileCols << ") pixels"
        << pyre::journal::endl;

    // launch the SAT kernel
    _sat <<<B,T>>> (dArena, tileRows, tileCols, dSAT);
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
            << "while launching the sum area tables kernel: "
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
            << "while computing the sum area tables: "
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
_sat(const value_t * dArena, std::size_t tileRows, std::size_t tileCols, value_t * dSAT)
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

    // the number of live workers we need
    auto workers = max(tileRows, tileCols);
    // excess workers
    if (t >= workers) {
        // can be sent home
        return;
    }

    // the shape of the SAT
    auto satRows = tileRows + 1;
    auto satCols = tileCols + 1;

    // the implementation here has three phases:
    //
    // - first, we take care of the border: every thread zeroes out one slot in the topmost row
    //   and one slot on the leftmost column
    // - next, each thread sweeps across a given row computing a running sum of the
    //   corresponding entries in the amplitude tile
    // - finally, each thread runs down each column setting up a running sum of its entries
    //
    // done carefully to make sure that only workers that have work to do are activated

    // point to the SAT my block is responsible for
    auto sat =  dSAT + b*satRows*satCols;
    // similarly, point to the start of my tile
    auto tile = dArena + b*tileRows*tileCols;
    // N.B.: any use of {dSAT} or {dArena} below this point is probably a bug...

    // get a handle to this thread group so we can synchronize the phases
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();

    // thread zero writes the upper left hand corner of the SAT
    if (t == 0) {
        sat[0] = 0;
    }
    // now, everybody
    if (t < tileCols) {
        // zeroes out a slot in the topmost row of the SAT; thread {t} takes care of the slot
        // at (0, t+1)
        sat[t+1] = 0;
    }
    // followed by everybody
    if (t < tileRows) {
        // zeroing out a slot in the leftmost column of the SAT; thread {t} takes care of the
        // slot at (t+1, 0)
        sat[(t+1)*satCols] = 0;
    }

    // barrier: make sure everybody is done updating the SAT
    cta.sync();

    // pick as many workers as there are rows in the tile
    if (t < tileRows) {
        // have each one find the tile row it will read from
        const value_t * read = tile + t*tileCols;
        // and the beginning of the sat row it will write to, starting with the row below the
        // border and skipping the zeroth entry in that row which belongs to the border
        value_t * write = sat + (t+1)*satCols + 1;
        // everybody initializes their running sum
        value_t sum = 0;
        // and run across their row
        for (int slot=0; slot<tileCols; ++slot) {
            // reading from the tile to update the running total
            sum += read[slot];
            // and writing to the corresponding spot in the sat
            write[slot] = sum;
        }
    }

    // barrier: make sure everybody is done updating their row
    cta.sync();

    // finally, pick as many workers as there are columns in a tile
    if (t < tileCols) {
        // initialize the running sum
        value_t sum = 0;
        // point to the beginning of its column after skipping over the border row and column;
        // we don't really need to skip the top row, since it is zero, but why pay for the
        // extra memory access only to avoid a tiny bit of pointer arithmetic? :)
        value_t * begin = sat + satCols + t + 1;
        // nobody is allowed to access memory past the end of this table
        value_t * end = sat + (satRows*satCols);
        // march down the column
        for (auto spot = begin; spot < end; spot += satCols) {
            // add the current value to the running total
            sum += *spot;
            // and store it in place
            *spot = sum;
        }
    }

    // all done
    return;
}


// end of file
