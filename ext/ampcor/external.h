// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_py_external_h)
#define ampcor_py_external_h


// STL
#include <complex>

// journal
#include <pyre/journal.h>

// pybind support
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>


// type aliases
namespace ampcor::py {
    // import {pybind11}
    namespace py = pybind11;
    // get the special {pybind11} literals
    using namespace py::literals;
}


# endif

// end of file
