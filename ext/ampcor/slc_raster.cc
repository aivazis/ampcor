// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2023 all rights reserved


// externals
#include "external.h"
// namespace setup
#include "forward.h"
// libampcor
#include <ampcor/dom.h>


// type aliases
namespace ampcor::py {
    using slc_raster_t = ampcor::dom::slc_raster_t;
}

// helpers
namespace ampcor::py {
    // the constructor
    inline auto slc_raster_constructor(py::tuple, py::object, bool) -> unique_pointer<slc_raster_t>;
}


// add bindings to SLC rasters
void
ampcor::py::slc_raster(py::module & m)
{
    // the SLC interface
    py::class_<slc_raster_t>(m, "SLCRaster")
        // constructor
        .def(
            // the constructor wrapper
            py::init([](py::tuple shape, py::object uri, bool create) {
                // ask the helper
                return slc_raster_constructor(shape, uri, create);
            }),
            // the signature
            "shape"_a, "uri"_a, "new"_a)

        // accessors
        // sizes of things: number of pixels
        .def_property_readonly(
            "cells",
            // the getter
            &slc_raster_t::cells,
            // the docstring
            "the number of pixels in the SLC")
        // sizes of things: memory footprint
        .def_property_readonly(
            "bytes",
            // the getter
            &slc_raster_t::bytes,
            // the docstring
            "the amount of memory occupied by this SLC, in bytes")

        // access to the shape
        .def_property_readonly(
            "tile",
            // the getter
            [](const slc_raster_t & slc) {
                // get the shape
                auto shape = slc.layout().shape();
                // convert it to a tuple
                auto pyShape = py::make_tuple(shape[0], shape[1]);
                // get the tile from {pyre.grid}
                auto pyFactory = py::module::import("pyre.grid").attr("tile");
                // invoke it
                auto pyTile = pyFactory("shape"_a = pyShape);
                // and return it
                return pyTile;
            },
            // the docstring
            "the shape of the SLC")

        // metamethods
        // data read access given an index
        .def(
            "__getitem__",
            // convert the incoming tuple into an index and fetch the data
            [](slc_raster_t & slc, py::tuple pyIdx) -> slc_raster_t::pixel_type & {
                // type aliases
                using index_t = slc_raster_t::index_type;
                using rank_t = slc_raster_t::index_type::rank_type;
                // make an index out of the python tuple
                slc_raster_t::index_type idx { pyIdx[0].cast<rank_t>(), pyIdx[1].cast<rank_t>() };
                // get the data and return it
                return slc[idx];
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
            // delegate directly to the {slc_raster_t}
            [](slc_raster_t & slc, size_t offset) -> slc_raster_t::pixel_type & {
                // easy enough
                return slc[offset];
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
ampcor::py::slc_raster_constructor(py::tuple pyShape, py::object pyURI, bool create)
    -> unique_pointer<slc_raster_t>
{
    // extract the shape
    int lines = py::int_(pyShape[0]);
    int samples = py::int_(pyShape[1]);

    // make a shape
    slc_raster_t::shape_type shape { lines, samples };
    // turn it into a layout
    slc_raster_t::layout_type layout { shape };
    // make a product specification
    slc_raster_t::spec_type spec { layout };

    // convert the path-like object into a string
    // get {os.fspath}
    auto fspath = py::module::import("os").attr("fspath");
    // call it and convert its return value into a string
    string_t filename = py::str(fspath(pyURI));

    // if we are supposed to create a new one
    if (create) {
        // build the product and return it
        return std::unique_ptr<slc_raster_t>(new slc_raster_t(spec, filename, spec.cells()));
    }

    // otherwise, just open an existing one in read/write mode
    return std::unique_ptr<slc_raster_t>(new slc_raster_t(spec, filename, true));
}


// end of file
