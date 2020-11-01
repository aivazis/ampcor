// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//

// code guard
#if !defined(ampcor_libampcor_cuda_correlators_Sequential_h)
#define ampcor_libampcor_cuda_correlators_Sequential_h


// resource management and orchestration of the execution of the correlation plan
template <class slcT, class offsetsT>
class ampcor::cuda::correlators::Sequential {
    // types
public:

    // metamethods
public:
    // destructor
    inline ~Sequential();
    // constructor
    inline
    Sequential(int rank,
               slc_const_reference ref, slc_const_reference sec,
               slc_shape_const_reference chip, slc_shape_const_reference window,
               int refineFactor, int refineMargin, int zoomFactor);


    // interface
public:

    // implementation
private:

    // data
private:
    // my rank
    int _rank;
    // the input rasters
    slc_const_reference _ref;
    slc_const_reference _sec;
    // the output raster
    offsets_reference _map;

    // the shape of the tiles
    const slc_shape_type _refShape;
    const slc_shape_type _secShape;

    // the correlation surface refinement parameters
    const int _refineFactor;
    const int _refineMargin;
    const int _zoomFactor;

    // disabled methods
private:
    // constructors
    Sequential(const Sequential &) = delete;
    Sequential(Sequential &&) = delete;
    Sequential & operator=(const Sequential &) = delete;
    Sequential & operator=(Sequential &&) = delete;
};


// the inline definitions
// sequential
#define ampcor_cuda_correlators_Sequential_icc
#include "Sequential.icc"
#undef ampcor_cuda_correlators_Sequential_icc


// code guard
#endif

// end of file
