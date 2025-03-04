// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2025 all rights reserved


// externals
#include "external.h"
// namespace setup
#include "forward.h"
// libampcor
#include <ampcor/dom.h>

// type aliases
namespace ampcor::py {
    using arena_t = ampcor::dom::arena_t<float>;
    using arena_const_raster_t = ampcor::dom::arena_const_raster_t<float>;

    using arena_layout_t = arena_const_raster_t::layout_type;
    using arena_index_t = arena_const_raster_t::index_type;
    using arena_shape_t = arena_const_raster_t::shape_type;
}

// helpers
namespace ampcor::py {
    // the constructor
    inline auto arena_const_raster_constructor(const arena_t &, py::object)
        -> unique_pointer<arena_const_raster_t>;
}


// add bindings to arena rasters
void
ampcor::py::arena_const_raster(py::module & m)
{
    // the arena interface
    py::class_<arena_const_raster_t>(m, "ArenaConstRaster")
        // constructor
        .def(
            // the constructor wrapper
            py::init([](const arena_t & spec, py::object uri) {
                return arena_const_raster_constructor(spec, uri);
            }),
            // the signature
            "spec"_a, "uri"_a)

        // accessors
        // sizes of things: number of pixels
        .def_property_readonly(
            "cells",
            // the getter
            &arena_const_raster_t::cells,
            // the docstring
            "the number of pixels in the arena")
        // sizes of things: memory footprint
        .def_property_readonly(
            "bytes",
            // the getter
            &arena_const_raster_t::bytes,
            // the docstring
            "the amount of memory occupied by this arena, in bytes")
        // access to the shape
        .def_property_readonly(
            "tile",
            // the getter
            [](const arena_const_raster_t & arena) {
                // get the shape
                auto shape = arena.layout().shape();
                // convert it to a tuple
                auto pyShape = py::make_tuple(shape[0], shape[1], shape[2]);
                // get the tile from {pyre.grid}
                auto pyFactory = py::module::import("pyre.grid").attr("tile");
                // invoke it
                auto pyTile = pyFactory("shape"_a = pyShape);
                // and return it
                return pyTile;
            },
            // the docstring
            "the shape of the arena")

        // metamethods
        // data read access given a native index
        .def(
            "__getitem__",
            // convert the incoming tuple into an index and fetch the data
            [](const arena_const_raster_t & arena, const arena_index_t & idx) {
                // get the data and return it
                return arena[idx];
            },
            // the signature
            "index"_a,
            // the docstring
            "access the data at the given index")
        // data read access given a python tuple
        .def(
            "__getitem__",
            // convert the incoming tuple into an index and fetch the data
            [](const arena_const_raster_t & arena, py::tuple pyIdx) {
                // type aliases
                using index_t = arena_const_raster_t::index_type;
                using rank_t = arena_const_raster_t::index_type::rank_type;
                // make an index out of the python tuple
                index_t idx { pyIdx[0].cast<rank_t>(), pyIdx[1].cast<rank_t>(),
                              pyIdx[2].cast<rank_t>() };
                // get the data and return it
                return arena[idx];
            },
            // the signature
            "index"_a,
            // the docstring
            "access the data at the given index")
        // data read access given an offset
        .def(
            "__getitem__",
            // delegate directly to the {arena_const_raster_t}
            [](const arena_const_raster_t & arena, size_t offset) {
                // easy enough
                return arena[offset];
            },
            // the signature
            "offset"_a,
            // the docstring
            "access the data at the given offset")
        // done
        ;

    // all done
    return;
}


// helper definitions
auto
ampcor::py::arena_const_raster_constructor(const arena_t & spec, py::object pyURI)
    -> unique_pointer<arena_const_raster_t>
{
    // convert the path-like object into a string
    // get {os.fspath}
    auto fspath = py::module::import("os").attr("fspath");
    // call it and convert its return value into a string
    string_t filename = py::str(fspath(pyURI));

    // build the product and return it
    return std::unique_ptr<arena_const_raster_t>(new arena_const_raster_t(spec, filename));
}


// end of file
