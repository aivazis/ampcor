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
    using arena_t = ampcor::dom::arena_t<float>;
}


// helpers
namespace ampcor::py {
    // the constructor
    inline auto
    arena_constructor(py::tuple, py::tuple) -> arena_t;
}


// add bindings to the arena product spec
void
ampcor::py::
arena(py::module &m) {
    // declare the product specification class
    auto pyArena = py::class_<arena_t>(m, "Arena");

    // add the arena interface
    pyArena
        // constructor
        .def(
             // the constructor wrapper
             py::init([](py::tuple pyOrigin, py::tuple pyShape) {
                 // get the helper to do its thing
                 return arena_constructor(pyOrigin, pyShape);
             }),
             // the signature
             "origin"_a, "shape"_a
             )

        // accessors
        // the size of a pixel
        .def_property_readonly("bytesPerPixel",
                               // the getter
                               [](const arena_t &) {
                                   // easy enough
                                   return sizeof(arena_t::pixel_type);
                               },
                               // the docstring
                               "memory footprint of an arena pixel, in bytes"
                               )

        // access to the layout
        .def_property_readonly("layout",
                               // the getter
                               &arena_t::layout,
                               // the docstring
                               "the layout of the arena raster"
                               )

        // and the shape
        .def_property_readonly("shape",
                               // the getter
                               [](const arena_t spec) {
                                   // easy enough
                                   return spec.layout().shape();
                               },
                               // the docstring
                               "the shape of the arena raster"
                               )

        // sizes of things: number of pixels
        .def_property_readonly("cells",
                               // the getter
                               &arena_t::cells,
                               // the docstring
                               "the number of pixels in the arena"
                               )
        // sizes of things: memory footprint
        .def_property_readonly("bytes",
                               // the getter
                               &arena_t::bytes,
                               // the docstring
                               "the amount of memory occupied by this arena, in bytes"
                               )
        // make a slice
        .def("slice",
             //
             [](const arena_t & arena, py::tuple pyOrigin, py::tuple pyShape) {
                 // type alises
                 using index_t = arena_t::layout_type::index_type;
                 using shape_t = arena_t::layout_type::shape_type;

                 // build the index
                 index_t idx { pyOrigin[0].cast<int>(), pyOrigin[1].cast<int>() };
                 // build the shape
                 shape_t shp { pyShape[0].cast<size_t>(), pyShape[1].cast<size_t>() };

                 // all done
                 return arena.layout().box(idx, shp);
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
arena_constructor(py::tuple pyOrigin, py::tuple pyShape) -> arena_t
{
    // extract the shape
    int pairs = py::int_(pyShape[0]);
    int xShape = py::int_(pyShape[1]);
    int yShape = py::int_(pyShape[2]);
    // and convert
    arena_t::layout_type::shape_type shape { pairs, xShape, yShape};

    // extract the origin
    int pOrig = py::int_(pyOrigin[0]);
    int xOrig = py::int_(pyOrigin[1]);
    int yOrig = py::int_(pyOrigin[2]);

    // and convert
    arena_t::layout_type::index_type origin { pOrig, xOrig, yOrig };

    // turn into a layout
    arena_t::layout_type layout { shape, origin };

    // make a product specification and return it
    return arena_t { layout };
}


// end of file
