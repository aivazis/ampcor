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
             py::init([](py::tuple shape, py::object uri, size_t cells) {
                          // ask the helper
                          return constructor(shape, uri, cells);
                      }),
             // the signature
             "shape"_a, "uri"_a, "cells"_a
             )
        // size of things
        // number of pixels
        .def_property_readonly("cells",
                      // the getter
                      &slc_t::cells,
                      // the docstring
                      "the number of pixels in the SLC"
                      )
        // memory footprint
        .def_property_readonly("bytes",
                      // the getter
                      &slc_t::bytes,
                      // the docstring
                      "the amount of memory occupied by this SLC, in bytes"
                      )
        // the static interface
        // the size of a pixel in bytes
        .def_property_readonly_static("bytesPerCell",
                                      // the getter
                                      [] (py::object) -> size_t {
                                          return slc_t::bytesPerCell();
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
constructor(py::tuple pyShape, py::object pyURI, size_t cells) -> unique_pointer<slc_t>
{
    // extract the shape
    int lines = py::int_(pyShape[0]);
    int samples = py::int_(pyShape[1]);
    // make a shape
    slc_t::spec_type::shape_type shape {lines, samples};
    // make a product specification out of the shape
    slc_t::spec_type spec { shape };

    // convert the path-like object into a string
    // get {os.fspath}
    auto fspath = py::module::import("os").attr("fspath");
    // call it and convert its return value into a string
    string_t filename = py::str(fspath(pyURI));

    // build the product and return it
    return std::unique_ptr<slc_t>(new slc_t(spec, filename, cells));
}


// end of file
