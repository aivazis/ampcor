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
    using offsets_const_raster_t = ampcor::dom::offsets_const_raster_t;
}

// helpers
namespace ampcor::py {
    // the constructor
    inline auto
    offsets_const_raster_constructor(py::tuple, py::object)
         -> unique_pointer<offsets_const_raster_t>;
}


// add bindings to offset maps
void
ampcor::py::
offsets_const_raster(py::module &m) {
    // the Offsets interface
    py::class_<offsets_const_raster_t>(m, "OffsetsConstRaster")
        // constructor
        .def(
             // the constructor wrapper
             py::init([](py::tuple shape, py::object uri) {
                          return offsets_const_raster_constructor(shape, uri);
                      }),
            // the signature
            "shape"_a, "uri"_a
            )

        // accessors
        // sizes of things: number of pixels
        .def_property_readonly("cells",
                      // the getter
                      &offsets_const_raster_t::cells,
                      // the docstring
                      "the number of pixels in the offset map"
                      )
        // sizes of things: memory footprint
        .def_property_readonly("bytes",
                      // the getter
                      &offsets_const_raster_t::bytes,
                      // the docstring
                      "the amount of memory occupied by this map, in bytes"
                      )

        // metamethods
        // data read access given an index
        .def("__getitem__",
             // convert the incoming tuple into an index and fetch the data
             [](const offsets_const_raster_t & map, py::tuple pyIdx) {
                 // type aliases
                 using index_t = offsets_const_raster_t::index_type;
                 using rank_t = offsets_const_raster_t::index_type::rank_type;
                 // make an index out of the python tuple
                 index_t idx {pyIdx[0].cast<rank_t>(), pyIdx[1].cast<rank_t>()};
                 // get the data and return it
                 return map[idx];
             },
             // the signature
             "index"_a,
             // the docstring
             "access the data at the given index"
             )
        // data read access given an offset
        .def("__getitem__",
             // delegate directly to the {offsets_const_raster_t}
             [](const offsets_const_raster_t & map, size_t offset) {
                 // easy enough
                 return map[offset];
             },
             // the signature
             "offset"_a,
             // the docstring
             "access the data at the given offset"
             )
        // done
        ;

    // all done
    return;
}


// helper definitions
auto
ampcor::py::
offsets_const_raster_constructor(py::tuple pyShape, py::object pyURI)
    -> unique_pointer<offsets_const_raster_t>
{
    // extract the shape
    int rows = py::int_(pyShape[0]);
    int cols = py::int_(pyShape[1]);

    // make a shape
    offsets_const_raster_t::shape_type shape { rows, cols };
    // turn it into a layout
    offsets_const_raster_t::layout_type layout { shape };
    // make a product specification
    offsets_const_raster_t::spec_type spec { layout };

    // convert the path-like object into a string
    // get {os.fspath}
    auto fspath = py::module::import("os").attr("fspath");
    // call it and convert its return value into a string
    string_t filename = py::str(fspath(pyURI));

    // build the product and return it
    return std::unique_ptr<offsets_const_raster_t>(new offsets_const_raster_t(spec, filename));
}


// end of file
