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
    // the worker
    template <class slcT = dom::slc_const_raster_t, class offsetsT = dom::offsets_raster_t>
    using sequential_t = Sequential<slcT, offsetsT>;
}


# endif

// end of file
