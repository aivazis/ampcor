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
    using offsets_raster_t = ampcor::dom::offsets_raster_t;
}

// helpers
namespace ampcor::py {
    // the constructor
    inline auto offsets_raster_constructor(py::tuple, py::object, bool)
        -> unique_pointer<offsets_raster_t>;
}


// add bindings to offset maps
void
ampcor::py::offsets_raster(py::module & m)
{
    // the Offsets interface
    py::class_<offsets_raster_t>(m, "OffsetsRaster")
        // constructor
        .def(
            // the constructor wrapper
            py::init([](py::tuple shape, py::object uri, bool create) {
                // ask the helper
                return offsets_raster_constructor(shape, uri, create);
            }),
            // the signature
            "shape"_a, "uri"_a, "new"_a)

        // accessors
        // sizes of things: number of pixels
        .def_property_readonly(
            "cells",
            // the getter
            &offsets_raster_t::cells,
            // the docstring
            "the number of pixels in the offset map")
        // sizes of things: memory footprint
        .def_property_readonly(
            "bytes",
            // the getter
            &offsets_raster_t::bytes,
            // the docstring
            "the amount of memory occupied by this map, in bytes")

        // metamethods
        // data read access given an index
        .def(
            "__getitem__",
            // convert the incoming tuple into an index and fetch the data
            [](offsets_raster_t & map, py::tuple pyIdx) -> offsets_raster_t::pixel_type & {
                // type aliases
                using index_t = offsets_raster_t::index_type;
                using rank_t = offsets_raster_t::index_type::rank_type;
                // make an index out of the python tuple
                index_t idx { pyIdx[0].cast<rank_t>(), pyIdx[1].cast<rank_t>() };
                // get the data and return it
                return map[idx];
            },
            // the signature
            "index"_a,
            // the docstring
            "access the data at the given index",
            // grant write access
            py::return_value_policy::reference)
        // data read access given an offset
        .def(
            "__getitem__",
            // delegate directly to the {offsets_raster_t}
            [](offsets_raster_t & map, size_t offset) -> offsets_raster_t::pixel_type & {
                // easy enough
                return map[offset];
            },
            // the signature
            "offset"_a,
            // the docstring
            "access the data at the given offset",
            // grant write access
            py::return_value_policy::reference)
        // done
        ;

    // all done
    return;
}


// helper definitions
auto
ampcor::py::offsets_raster_constructor(py::tuple pyShape, py::object pyURI, bool create)
    -> unique_pointer<offsets_raster_t>
{
    // extract the shape
    int rows = py::int_(pyShape[0]);
    int cols = py::int_(pyShape[1]);
    // make a shape
    offsets_raster_t::shape_type shape { rows, cols };
    // turn it into a layout
    offsets_raster_t::layout_type layout { shape };
    // make a product specification
    offsets_raster_t::spec_type spec { layout };

    // convert the path-like object into a string
    // get {os.fspath}
    auto fspath = py::module::import("os").attr("fspath");
    // call it and convert its return value into a string
    string_t filename = py::str(fspath(pyURI));

    // check whether we are supposed to create a new one, or open an existing one
    auto product = create ? new offsets_raster_t(spec, filename, spec.cells()) :
                            new offsets_raster_t(spec, filename, true);
    // wrap it and return it
    return std::unique_ptr<offsets_raster_t>(product);
}


// end of file
