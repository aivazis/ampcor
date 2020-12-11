// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//

// code guard
#if !defined(ampcor_libampcor_cuda_correlators_kernels_h)
#define ampcor_libampcor_cuda_correlators_kernels_h


// externals
#include <complex>
#include <cuComplex.h>


// forward declarations
namespace ampcor::cuda::kernels {
    // compute the correlation matrix
    void correlate(const float * refArena, const float * refStats,
                   const float * secArena, const float * secStats,
                   std::size_t pairs,
                   std::size_t refRows, std::size_t refCols,
                   std::size_t secRows, std::size_t secCols,
                   std::size_t corRows, std::size_t corCols,
                   float * dCorrelation);

    // remove phase ramps
    void deramp(std::complex<float> * arena,
                std::size_t pairs, std::size_t arenaRows, std::size_t arenaCols,
                std::size_t tileRows, std::size_t tileCols);

    // compute amplitudes of the tile pixels
    void detect(const std::complex<float> * cArena,
                std::size_t pairs, std::size_t rows, std::size_t cols,
                float * rArena);

    // compute the locations of the maximum value of the correlation map
    void maxcor(const float * cor,
                int pairs, int corRows, int corCols,
                int orgRow, int orgCol, float zoomFactor,
                float * loc);

    // subtract the tile mean from each reference pixel
    void refStats(float * rArena,
                  std::size_t pairs, std::size_t refRows, std::size_t refCols,
                  float * stats);

    // build the sum area tables for the secondary tiles
    void sat(const float * rArena,
             std::size_t pairs, std::size_t tileRows, std::size_t tileCols,
             float * sat);

    // compute the average amplitude for all possible placements of a reference shape
    // within the search windows
    void secStats(const float * sat,
                  std::size_t pairs,
                  std::size_t refRows, std::size_t refCols,
                  std::size_t secRows, std::size_t secCols,
                  std::size_t corRows, std::size_t corCols,
                  float * stats);

    // spread the spectrum, a necessary step while interpolating using FFTs
    void spread(std::complex<float> * arena,
                std::size_t pairs, std::size_t arenaRows, std::size_t arenaCols,
                std::size_t tileRows, std::size_t tileCols);
}


// code guard
#endif

// end of file
