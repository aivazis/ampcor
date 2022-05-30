// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2022 all rights reserved


// external dependencies
#include "external.h"
// namespace setup
#include "forward.h"


// the module entry point
PYBIND11_MODULE(ampcor, m)
{
    // the doc string
    m.doc() = "the libampcor bindings";

    // bind the opaque types
    ampcor::py::opaque(m);
    // register the exception types
    ampcor::py::exceptions(m);

    // slc
    ampcor::py::slc(m);
    ampcor::py::slc_raster(m);
    ampcor::py::slc_const_raster(m);

    // offset maps
    ampcor::py::offsets(m);
    ampcor::py::offsets_raster(m);
    ampcor::py::offsets_const_raster(m);

    // arenas
    ampcor::py::arena(m);
    ampcor::py::arena_const_raster(m);

    // plans
    ampcor::py::plan(m);
    // and correlators
    ampcor::py::sequential(m);

    // viz
    ampcor::py::viz(m);
}


// end of file
