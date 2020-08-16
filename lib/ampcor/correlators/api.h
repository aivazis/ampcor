// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_correlators_api_h)
#define ampcor_correlators_api_h


// publicly visible types
namespace ampcor::correlators {
    // the sequential worker
    template <class slcT = dom::slc_const_raster_t, class offsetsT = dom::offsets_raster_t>
    using sequential_t = Sequential<slcT, offsetsT>;

    // the threaded worker
    template <class sequentialT>
    using threaded_t = Threaded<sequentialT>;
}


#endif

// end of file
