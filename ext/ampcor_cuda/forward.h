// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2021 all rights reserved

// code guard
#if !defined(ampcor_cuda_py_forward_h)
#define ampcor_cuda_py_forward_h


// the {ampcor} namespace
namespace ampcor::cuda::py {
    // bindings of opaque types
    void opaque(py::module &);
    // exceptions
    void exceptions(py::module &);

    // access to the correlators
    void sequential(py::module &);
}


#endif

// end of file
