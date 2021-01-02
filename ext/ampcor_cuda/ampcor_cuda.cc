// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2021 all rights reserved


// external dependencies
#include "external.h"
// namespace setup
#include "forward.h"


// the module entry point
PYBIND11_MODULE(ampcor_cuda, m) {
    // the doc string
    m.doc() = "the libampcor_cuda bindings";

    // bind the opaque types
    ampcor::cuda::py::opaque(m);
    // register the exception types
    ampcor::cuda::py::exceptions(m);

    // correlators
    ampcor::cuda::py::sequential(m);
}


// end of file
