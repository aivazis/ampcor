// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2025 all rights reserved

// code guard
#if !defined(ampcor_py_external_h)
#define ampcor_py_external_h


// STL
#include <complex>
#include <string>
#include <vector>

// journal
#include <pyre/journal.h>
// timers
#include <pyre/timers.h>

// pybind support
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// libampcor
#include <ampcor/dom.h>
#include <ampcor/correlators.h>
#include <ampcor/viz.h>


// type aliases
namespace ampcor::py {
    // import {pybind11}
    namespace py = pybind11;
    // get the special {pybind11} literals
    using namespace py::literals;

    // sizes of things
    using size_t = std::size_t;
    // strings
    using string_t = std::string;

    // pointer wrappers
    template <class T>
    using unique_pointer = std::unique_ptr<T>;

    using proctimer_t = pyre::timers::process_timer_t;
}


#endif

// end of file
