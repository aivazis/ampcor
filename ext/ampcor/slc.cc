// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2022 all rights reserved


// externals
#include "external.h"
// namespace setup
#include "forward.h"
// libampcor
#include <ampcor/dom.h>


// type aliases
namespace ampcor::py {
    // the product spec
    using slc_t = ampcor::dom::slc_t;
}


// helpers
namespace ampcor::py {
    // the constructor
    inline auto slc_constructor(py::tuple) -> slc_t;
}


// add bindings to the SLC product spec
void
ampcor::py::slc(py::module & m)
{
    // declare the product specification class
    auto pySLC = py::class_<slc_t>(m, "SLC");

    // add the SLC interface
    pySLC
        // constructor
        .def(
            // the constructor wrapper
            py::init([](py::tuple pyShape) {
                // get the helper to do its thing
                return slc_constructor(pyShape);
            }),
            // the signature
            "shape"_a)

        // accessors
        // the size of a pixel
        .def_property_readonly_static(
            // name
            "bytesPerPixel",
            // the getter
            [](py::object) {
                // easy enough
                return sizeof(slc_t::pixel_type);
            },
            // the docstring
            "memory footprint of an SLC pixel, in bytes")

        // access to the layout
        .def_property_readonly(
            "layout",
            // the getter
            &slc_t::layout,
            // the docstring
            "the layout of the SLC raster")

        // and the shape
        .def_property_readonly(
            "shape",
            // the getter
            [](const slc_t spec) {
                // easy enough
                return spec.layout().shape();
            },
            // the docstring
            "the shape of the SLC raster")

        // sizes of things: number of pixels
        .def_property_readonly(
            "cells",
            // the getter
            &slc_t::cells,
            // the docstring
            "the number of pixels in the SLC")
        // sizes of things: memory footprint
        .def_property_readonly(
            "bytes",
            // the getter
            &slc_t::bytes,
            // the docstring
            "the amount of memory occupied by this SLC, in bytes")
        // make a slice
        .def(
            "slice",
            //
            [](const slc_t & slc, py::tuple pyOrigin, py::tuple pyShape) {
                // type alises
                using index_t = slc_t::layout_type::index_type;
                using shape_t = slc_t::layout_type::shape_type;

                // build the index
                index_t idx { pyOrigin[0].cast<int>(), pyOrigin[1].cast<int>() };
                // build the shape
                shape_t shp { pyShape[0].cast<size_t>(), pyShape[1].cast<size_t>() };

                // all done
                return slc.layout().box(idx, shp);
            },
            // the docstring
            "make a slice at {origin} with the given {shape}",
            // the signature
            "origin"_a, "shape"_a)
        // done
        ;

    // all done
    return;
}


// helper definitions
auto
ampcor::py::slc_constructor(py::tuple pyShape) -> slc_t
{
    // extract the shape
    int lines = py::int_(pyShape[0]);
    int samples = py::int_(pyShape[1]);

    // make a shape
    slc_t::layout_type::shape_type shape { lines, samples };
    // turn into a layout
    slc_t::layout_type layout { shape };

    // make a product specification and return it
    return slc_t { layout };
}


// end of file
