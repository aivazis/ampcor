// -*- C++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_cuda_correlators_api_h)
#define ampcor_cuda_correlators_api_h


// publicly visible types
namespace ampcor::cuda::correlators {
    // make a selector
    template <typename T, bool isReadOnly>
    using cudaheap_t = CUDAHeap<T, isReadOnly>;

    // device memory
    template <typename T>
    using devmem_t = CUDAHeap<T, false>;
    // read-only version
    template <typename T>
    using constdevmem_t = CUDAHeap<T, true>;

    // a grid on the device
    template <typename specT, bool isReadOnly = true>
    using devgrid_t = pyre::grid::grid_t<typename specT::layout_type,
                                         cudaheap_t<typename specT::pixel_type, isReadOnly>>;

    // an arena on the device
    template <typename pixelT>
    using devarena_raster_t = devgrid_t<dom::arena_t<pixelT>, false>;

    // the worker
    template <class slcT = dom::slc_const_raster_t, class offsetsT = dom::offsets_raster_t>
    using sequential_t = Sequential<slcT, offsetsT>;
}


# endif

// end of file
