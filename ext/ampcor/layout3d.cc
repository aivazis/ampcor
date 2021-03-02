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
ampcor::py::layout3d(py::module & m)
{
    // 3d layouts
    auto layoutCls = py::class_<layout3d_t>(m, "Layout3D");

    // constructors
    // from a shape
    layoutCls.def(
        // the handler
        py::init<const shape3d_t &>(),
        // the signature
        "shape"_a);

    // from a shape and an origin
    layoutCls.def(
        // the handler
        py::init<const shape3d_t &, const index3d_t &>(),
        // the signature
        "shape"_a, "origin"_a);

    // accessors
    // my shape
    layoutCls.def_property_readonly(
        // the name
        "shape",
        // the getter
        &layout3d_t::shape,
        // the docstring
        "get my shape");

    // my origin
    layoutCls.def_property_readonly(
        // the name
        "origin",
        // the getter
        &layout3d_t::origin,
        // the docstring
        "get my origin");

    // my origin
    layoutCls.def_property_readonly(
        // the name
        "order",
        // the getter
        [](const layout3d_t & layout) { return layout.order(); },
        // the docstring
        "get my packing order");

    // my strides
    layoutCls.def_property_readonly(
        // the name
        "strides",
        // the getter
        &layout3d_t::strides,
        // the docstring
        "get my strides");

    // my nudge
    layoutCls.def_property_readonly(
        // the name
        "nudge",
        // the getter
        &layout3d_t::nudge,
        // the docstring
        "get my nudge");

    // sizes of things: number of pixels
    layoutCls.def_property_readonly(
        "cells",
        // the getter
        &layout3d_t::cells,
        // the docstring
        "the number of pixels in the arena");

    // methods
    layoutCls.def(
        "box",
        // the handler
        &layout3d_t::box,
        // the signature
        "origin"_a, "shape"_a,
        // the docstring
        "carve a portion of a layout");

    // indexing
    // get the index that corresponds to a given offset
    layoutCls.def(
        // the name
        "index",
        // the function
        &layout3d_t::index,
        // the signature,
        "offset"_a,
        // the docstring
        "get the index that corresponds to the given {offset}");
    // get the offset that corresponds to the given {index}
    layoutCls.def(
        // the name
        "offset",
        // the function
        &layout3d_t::offset,
        // the signature
        "index"_a,
        // the docstring
        "get the offset that corresponds to the given {index}");
    // same as above, with {index} a tuple
    layoutCls.def(
        // the name
        "offset",
        // the function
        [](const layout3d_t & layout, std::tuple<int, int> index) {
            // unpack
            auto [i0, i1] = index;
            // make an index
            index3d_t idx { i0, i1 };
            // and ask for the offset
            return layout.offset(idx);
        },
        // the signature
        "index"_a,
        // the docstring
        "get the offset that corresponds to the given {index}");

    // iteration support
    layoutCls.def(
        // the name
        "__iter__",
        // the implementation
        [](const layout3d_t & layout) { return py::make_iterator(layout.begin(), layout.end()); },
        // the docstring
        "iterate over the layout in its natural order",
        // make sure the shape lives long enough
        py::keep_alive<0, 1>());

    // rank
    layoutCls.def_property_readonly_static(
        // the name of the property
        "rank",
        // the getter
        [](py::object) { return layout3d_t::rank(); },
        // the docstring
        "the number of cells in this shape");

    // all done
    return;
}


// end of file
