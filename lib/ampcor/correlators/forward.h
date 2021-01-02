// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2021 all rights reserved

// code guard
#if !defined(ampcor_correlators_forward_h)
#define ampcor_correlators_forward_h


// set up the namespace
namespace ampcor::correlators {
    // the sequential worker
    template <class slcT, class offsetsT>
    class Sequential;

    // the threaded worker
    template <class sequentialT>
    class Threaded;
}


#endif

// end of file
