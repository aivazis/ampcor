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
namespace ampcor::py {
    // the product spec
    using slc_t = ampcor::dom::slc_t;
    // and its parts
    using slc_layout_t = slc_t::layout_type;
    using slc_index_t = slc_t::layout_type::index_type;
    using slc_shape_t = slc_t::layout_type::shape_type;
}


// helpers
namespace ampcor::py {
    // the constructor
    inline auto
    slc_constructor(py::tuple) -> slc_t;
}


// add bindings to the SLC product spec
void
ampcor::py::
slc(py::module &m) {
    // declare the product specification class
    auto pySLC = py::class_<slc_t>(m, "SLC");
    // embed its layout
    auto pySLCLayout = py::class_<slc_layout_t>(pySLC, "Layout", py::module_local());
    // its index
    auto pySLCIndex = py::class_<slc_index_t>(pySLC, "Index", py::module_local());
    // and its shape
    auto pySLCShape = py::class_<slc_shape_t>(pySLCLayout, "Shape", py::module_local());

    // add the SLC layout interface
    pySLCLayout
        .def_property_readonly("origin",
                               // the getter
                               &slc_layout_t::origin,
                               // the docstring
                               "the origin of the SLC layout"
                               )

        .def_property_readonly("shape",
                               // the getter
                               &slc_layout_t::shape,
                               // the docstring
                               "the shape of the SLC layout"
                               )

        // sizes of things: number of pixels
        .def_property_readonly("cells",
                               // the getter
                               &slc_layout_t::cells,
                               // the docstring
                               "the number of pixels in the SLC"
                               )
        // and number of bytes
        .def_property_readonly("bytes",
                               // the getter
                               [](const slc_layout_t & layout) {
                                   // easy enough
                                   return layout.cells() * sizeof(slc_t::pixel_type);
                               },
                               // the docstring
                               "the memory footprint of the SLC, in bytes"
                               )
        // done
        ;

    // add the SLC index interface
    pySLCIndex
        // access to individual ranks
        .def("__getitem__",
             // return the value of the requested rank
             [](const slc_index_t & index, int rank) {
                 return index[rank];
             },
             // signature
             "rank"_a
             )
        // iteration support
        .def("__iter__",
             // make an iterator and return it
             [](const slc_index_t & index) {
                 return py::make_iterator(index.begin(), index.end());
             },
             // make sure the index lives long enough
             py::keep_alive<0,1>()
             )
        // done
        ;

    // add the SLC shape interface
    pySLCShape
        // sizes of things: number of pixels
        .def_property_readonly("cells",
                               // the getter
                               &slc_shape_t::cells,
                               // the docstring
                               "the number of pixels in the SLC"
                               )
        // access to individual ranks
        .def("__getitem__",
             // return the value of the requested rank
             [](const slc_shape_t & shape, int idx) {
                 return shape[idx];
             },
             // signature
             "index"_a
             )
        // iteration support
        .def("__iter__",
             // make an iterator and return it
             [](const slc_shape_t & shape) {
                 return py::make_iterator(shape.begin(), shape.end());
             },
             // make sure the shape lives long enough
             py::keep_alive<0,1>()
             )
        // done
        ;

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
             "shape"_a
             )

        // accessors
        .def_property_readonly("layout",
                               // the getter
                               &slc_t::layout,
                               // the docstring
                               "the layout of the SLC raster"
                               )

        .def_property_readonly("shape",
                               // the getter
                               &slc_t::shape,
                               // the docstring
                               "the shape of the SLC raster"
                               )

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
        // make a slice
        .def("slice",
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
             "origin"_a, "shape"_a
             )
        // done
        ;

    // all done
    return;
}


// helper definitions
auto
ampcor::py::
slc_constructor(py::tuple pyShape) -> slc_t
{
    // extract the shape
    int lines = py::int_(pyShape[0]);
    int samples = py::int_(pyShape[1]);
    // make a shape
    slc_t::layout_type::shape_type shape {lines, samples};
    // turn it into a layout
    slc_t::layout_type layout { shape };
    // make a product specification out of the layout and return it
    return slc_t { layout };
}


// end of file
