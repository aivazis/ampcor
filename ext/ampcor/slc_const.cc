// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2020 all rights reserved


// externals
#include "external.h"
// namespace setup
#include "forward.h"


// grid + memory
#include <p2/grid.h>
// libampcor
#include <ampcor/dom.h>


// add bindings to SLC rasters
void
ampcor::py::
slc_const(py::module &m) {
    // the SLC interface
    py::class_<dom::slc_const_t>(m, "ConstSLC")
        // the static interface
        // the size of a pixel in bytes
        .def_property_readonly_static("pixelFootprint",
                                      // the getter
                                      [] (py::object) -> size_t {
                                          return dom::slc_const_t::pixelFootprint();
                                      },
                                      // the docstring
                                      "the size of an SLC pixel"
                                      )
        // done
        ;

    // all done
    return;
}


// end of file
