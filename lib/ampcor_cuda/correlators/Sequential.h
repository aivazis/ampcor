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
    // my template parameters
    using slc_type = slcT;
    using offsets_type = offsetsT;
    // typical uses
    using slc_const_reference = const slc_type &;
    using offsets_reference = offsets_type &;
    // the product specs
    using slc_spec = typename slc_type::spec_type;
    using offsets_spec = typename offsets_type::spec_type;

    // the slc pixel base type
    using slc_value_type = typename slc_spec::value_type;
    // the slc pixel type
    using slc_pixel_type = typename slc_spec::pixel_type;

    // my constructor and implementation depend on the layout of input tiles
    using slc_layout_type = typename slc_spec::layout_type;
    using slc_shape_type = typename slc_layout_type::shape_type;
    using slc_index_type = typename slc_layout_type::index_type;
    using slc_layout_const_reference = const slc_layout_type &;
    using slc_shape_const_reference = const slc_shape_type &;
    using slc_index_const_reference = const slc_index_type &;

    // the specification of the portion of the work assigned to me uses output tiles
    using offsets_layout_type = typename offsets_spec::layout_type;
    using offsets_shape_type = typename offsets_layout_type::shape_type;
    using offsets_index_type = typename offsets_layout_type::index_type;
    using offsets_layout_const_reference = const offsets_layout_type &;
    using offsets_shape_const_reference = const offsets_shape_type &;
    using offsets_index_const_reference = const offsets_index_type &;

    // metamethods
public:
    // destructor
    inline ~Sequential();
    // constructor
    inline
    Sequential(int rank,
               slc_const_reference ref, slc_const_reference sec, offsets_reference map,
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
#define ampcor_cuda_correlators_Sequential_icc
#include "Sequential.icc"
#undef ampcor_cuda_correlators_Sequential_icc


// code guard
#endif

// end of file
