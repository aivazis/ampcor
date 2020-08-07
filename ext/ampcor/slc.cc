// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2020 all rights reserved


// externals
#include "external.h"
// namespace setup
#include "forward.h"


// libampcor
#include <ampcor/dom.h>


// type aliases
using slc_t = ampcor::dom::slc_t;


// add bindings to the SLC product spec
void
ampcor::py::
slc(py::module &m) {
    // the SLC interface
    py::class_<slc_t>(m, "SLC")
        // constructor
        .def(
             // the constructor wrapper
             py::init([](py::tuple pyShape) {
                          // extract the shape
                          int lines = py::int_(pyShape[0]);
                          int samples = py::int_(pyShape[1]);
                          // make a shape
                          slc_t::layout_type::shape_type shape {lines, samples};
                          // turn it into a layout
                          slc_t::layout_type layout { shape };
                          // make a product specification out of the layout and return it
                          return slc_t { layout };
                      }),
             // the signature
             "shape"_a
             )

        // accessors
        // sizes of things: number of pixels
        .def_property_readonly("cells",
                      // the getter
                      &slc_t::cells,
                      // the docstring
                      "the number of pixels in the SLC"
                      )
        // sizes of things: memory footprint
        .def_property_readonly("bytes",
                      // the getter
                      &slc_t::bytes,
                      // the docstring
                      "the amount of memory occupied by this SLC, in bytes"
                      )
        // done
        ;

    // all done
    return;
}


// end of file
