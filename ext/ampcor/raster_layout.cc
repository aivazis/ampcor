// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2021 all rights reserved


// externals
#include "external.h"
// namespace setup
#include "forward.h"
// libampcor
#include <ampcor/dom.h>


// type aliases
namespace ampcor::py {
    // the layout parts
    using raster_layout_t = dom::layout_t<2>;
    using raster_index_t = raster_layout_t::index_type;
    using raster_shape_t = raster_layout_t::shape_type;
}


// add bindings for the raster layout parts
void
ampcor::py::
raster_layout(py::module &m) {
    // the raster layout
    auto pyRasterLayout = py::class_<raster_layout_t>(m, "RasterLayout");
    // its index
    auto pyRasterIndex = py::class_<raster_index_t>(m, "RasterIndex");
    // and its shape
    auto pyRasterShape = py::class_<raster_shape_t>(m, "RasterShape");

    // add the raster layout interface
    pyRasterLayout
        // properties
        .def_property_readonly("origin",
                               // the getter
                               &raster_layout_t::origin,
                               // the docstring
                               "the origin of the raster layout"
                               )

        .def_property_readonly("shape",
                               // the getter
                               &raster_layout_t::shape,
                               // the docstring
                               "the shape of the raster layout"
                               )

        // sizes of things: number of pixels
        .def_property_readonly("cells",
                               // the getter
                               &raster_layout_t::cells,
                               // the docstring
                               "the number of pixels in the raster"
                               )
        // methods
        .def("box",
             // the handler
             &raster_layout_t::box,
             // the signature
             "origin"_a, "shape"_a,
             // the docstring
             "carve a portion of a layout"
             )
        // done
        ;

    // add the raster index interface
    pyRasterIndex
        // access to individual ranks
        .def("__getitem__",
             // return the value of the requested rank
             [](const raster_index_t & index, int rank) {
                 return index[rank];
             },
             // signature
             "rank"_a
             )
        // iteration support
        .def("__iter__",
             // make an iterator and return it
             [](const raster_index_t & index) {
                 return py::make_iterator(index.begin(), index.end());
             },
             // make sure the index lives long enough
             py::keep_alive<0,1>()
             )
        // done
        ;

    // add the raster shape interface
    pyRasterShape
        // sizes of things: number of pixels
        .def_property_readonly("cells",
                               // the getter
                               &raster_shape_t::cells,
                               // the docstring
                               "the number of pixels in the raster"
                               )
        // access to individual ranks
        .def("__getitem__",
             // return the value of the requested rank
             [](const raster_shape_t & shape, int idx) {
                 return shape[idx];
             },
             // signature
             "index"_a
             )
        // iteration support
        .def("__iter__",
             // make an iterator and return it
             [](const raster_shape_t & shape) {
                 return py::make_iterator(shape.begin(), shape.end());
             },
             // make sure the shape lives long enough
             py::keep_alive<0,1>()
             )
        // done
        ;

    // all done
    return;
}


// end of file
