// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2021 all rights reserved


// externals
#include "external.h"
// namespace setup
#include "forward.h"


// add bindings for the grid layouts used in this package
void
ampcor::py::shape3d(py::module & m)
{
    // the shape
    auto shapeCls = py::class_<shape3d_t>(m, "Shape3D");

    // populate {Shape3D}
    // constructor
    shapeCls.def(
        // convert python tuples into shapes
        py::init([](std::tuple<int, int> pyShape) {
            // unpack
            auto [s0, s1] = pyShape;
            // build a shape and return it
            return new layout3d_t::shape_type(s0, s1);
        }),
        // the signature
        "shape"_a);

    // rank
    shapeCls.def_property_readonly_static(
        // the name of the property
        "rank",
        // the getter
        [](py::object) { return shape3d_t::rank(); },
        // the docstring
        "the number of cells in this shape");

    // number of cells
    shapeCls.def_property_readonly(
        // the name of the property
        "cells",
        // the getter
        &shape3d_t::cells,
        // the docstring
        "the number of cells in this shape");

    // access to individual ranks
    shapeCls.def(
        // the name of the method
        "__getitem__",
        // the getter
        [](const shape3d_t & shape, int idx) { return shape[idx]; },
        // the signature
        "index"_a,
        // the docstring
        "get the value of a given rank");

    // iteration support
    shapeCls.def(
        // the name of the method
        "__iter__",
        // the implementation
        [](const shape3d_t & shape) { return py::make_iterator(shape.begin(), shape.end()); },
        // the docstring
        "iterate over the ranks",
        // make sure the shape lives long enough
        py::keep_alive<0, 1>());

    // string representation
    shapeCls.def(
        // the name of the method
        "__str__",
        // the implementation
        [](const shape3d_t & shape) {
            // make a buffer
            std::stringstream buffer;
            // inject my value
            buffer << shape;
            // and return the value
            return buffer.str();
        },
        // the docstring
        "generate a string representation");

    // all done
    return;
}


// end of file
