// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_py_forward_h)
#define ampcor_py_forward_h


// the {ampcor} namespace
namespace ampcor::py {
    // bindings of opaque types
    void opaque(py::module &);
    // exceptions
    void exceptions(py::module &);

    // access to SLC instances
    void slc(py::module &);
}


#endif

// end of file
