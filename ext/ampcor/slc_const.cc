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
    // type aliase
    using slc_t = dom::slc_const_t;

    // the SLC interface
    py::class_<slc_t>(m, "ConstSLC")

        // constructor
        .def(
             // the handler
             py::init([](py::tuple shape, std::string filename) {
                          // extract the shape
                          int lines = shape[0].cast<int>();
                          int samples = shape[1].cast<int>();
                          // make a spec out of the shape
                          slc_t::product_type spec { {lines, samples} };
                          // build the product
                          return std::unique_ptr<slc_t>(new slc_t(spec, filename));
                      }),
            // the signature
            "shape"_a, "filename"_a
            )
        // size of things
        // number of pixels
        .def_property_readonly("capacity",
                      // the getter
                      &slc_t::capacity,
                      // the docstring
                      "the number of pixels in the SLC"
                      )
        // memory footprint
        .def_property_readonly("footprint",
                      // the getter
                      &slc_t::footprint,
                      // the docstring
                      "the amount of memory occupied by this SLC, in bytes"
                      )
        // the static interface
        // the size of a pixel in bytes
        .def_property_readonly_static("pixelFootprint",
                                      // the getter
                                      [] (py::object) -> size_t {
                                          return slc_t::pixelFootprint();
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
