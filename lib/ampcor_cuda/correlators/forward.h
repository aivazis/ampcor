// -*- C++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2025 all rights reserved

// code guard
#if !defined(ampcor_cuda_correlators_forward_h)
#define ampcor_cuda_correlators_forward_h


// set up the namespace
namespace ampcor::cuda::correlators {
    // memory block on the device
    template <typename T, bool isConst> class CUDAHeap;

    // the correlators
    template <class slcT, class offsetsT>
    class Sequential;
}


# endif

// end of file
