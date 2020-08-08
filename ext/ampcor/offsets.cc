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
using offsets_t = ampcor::dom::offsets_t;


// add bindings to the Offsets product spec
void
ampcor::py::
offsets(py::module &m) {
    // the Offsets interface
    py::class_<offsets_t>(m, "Offsets")
        // constructor
        .def(
             // the constructor wrapper
             py::init([](py::tuple pyShape) {
                          // extract the shape
                          int lines = py::int_(pyShape[0]);
                          int samples = py::int_(pyShape[1]);
                          // make a shape
                          offsets_t::layout_type::shape_type shape {lines, samples};
                          // turn it into a layout
                          offsets_t::layout_type layout { shape };
                          // make a product specification out of the layout and return it
                          return offsets_t { layout };
                      }),
             // the signature
             "shape"_a
             )

        // accessors
        // sizes of things: number of cells
        .def_property_readonly("cells",
                      // the getter
                      &offsets_t::cells,
                      // the docstring
                      "the number of pixels in the offsets map"
                      )
        // sizes of things: memory footprint
        .def_property_readonly("bytes",
                      // the getter
                      &offsets_t::bytes,
                      // the docstring
                      "the amount of memory occupied by this offsets map, in bytes"
                      )
        // done
        ;

    // all done
    return;
}


// end of file
