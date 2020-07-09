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


// type aliases
using slc_t = ampcor::dom::slc_t;

// helpers
namespace ampcor::py {
    // the constructor
    inline auto
    constructor(py::tuple, py::object, size_t) -> unique_pointer<slc_t>;
}


// add bindings to SLC rasters
void
ampcor::py::
slc(py::module &m) {
    // the SLC interface
    py::class_<dom::slc_t>(m, "SLC")
        // constructor
        .def(
             // the constructor wrapper
             py::init([](py::tuple shape, py::object uri, size_t capacity) {
                          // ask the helper
                          return constructor(shape, uri, capacity);
                      }),
             // the signature
             "shape"_a, "uri"_a, "capacity"_a
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


// helper definitions
auto
ampcor::py::
constructor(py::tuple shape, py::object uri, size_t capacity) -> unique_pointer<slc_t>
{
    // extract the shape
    int lines = py::int_(shape[0]);
    int samples = py::int_(shape[1]);
    // make a product specification out of the shape
    slc_t::product_type spec { {lines, samples} };

    // convert the path-like object into a string
    // get {os.fspath}
    auto fspath = py::module::import("os").attr("fspath");
    // call it and convert its return value into a string
    string_t filename = py::str(fspath(uri));

    // build the product and return it
    return std::unique_ptr<slc_t>(new slc_t(spec, filename, capacity));
}


// end of file
