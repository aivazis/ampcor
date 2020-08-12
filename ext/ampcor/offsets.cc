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
    using offsets_t = ampcor::dom::offsets_t;
    using offsets_pixel_t = offsets_t::pixel_type;
    using offsets_layout_t = offsets_t::layout_type;
    using offsets_index_t = offsets_t::layout_type::index_type;
    using offsets_shape_t = offsets_t::layout_type::shape_type;
}


// helpers
namespace ampcor::py {
    // the spec constructor
    inline auto offsets_constructor(py::tuple) -> offsets_t;
}

// add bindings to the Offsets product spec
void
ampcor::py::
offsets(py::module &m) {
    // the product spec
    py::class_<offsets_t> pyOffsets(m, "Offsets");
    // its pixel type
    py::class_<offsets_pixel_t> pyPixel(pyOffsets, "Pixel", py::module_local());
    // embed its layout
    auto pyOffsetsLayout = py::class_<offsets_layout_t>(pyOffsets, "Layout", py::module_local());
    // its index
    auto pyOffsetsIndex = py::class_<offsets_index_t>(pyOffsets, "Index", py::module_local());
    // and its shape
    auto pyOffsetsShape = py::class_<offsets_shape_t>(pyOffsetsLayout, "Shape", py::module_local());

    // the pixel definition
    pyPixel
        // the reference index
        .def_property("ref",
                      // the getter
                      [](const offsets_pixel_t & pxl) {
                          return pxl.ref;
                      },
                      // the setter
                      [](offsets_pixel_t & pxl, std::pair<float, float> ref) {
                          pxl.ref = ref;
                      },
                      // the docstring
                      "the reference pixel indices"
                      )
        // the shift
        .def_property("delta",
                      // the getter
                      [](const offsets_pixel_t & pxl) {
                          return pxl.shift;
                      },
                      // the setter
                      [](offsets_pixel_t & pxl, std::pair<float, float> delta) {
                         pxl.shift = delta;
                      },
                      // the docstring
                      "the offset to the matching pixel in the secondary raster"
                      )
        // the level of confidence in the mapping
        .def_property("confidence",
                      // the getter
                      [](const offsets_pixel_t & pxl) {
                          return pxl.confidence;
                      },
                      // the setter
                      [](offsets_pixel_t & pxl, float confidence) {
                         pxl.confidence = confidence;
                      },
                      // the docstring
                      "the level of confidence in the mapping"
                      )
        // the signal to noise ration
        .def_property("snr",
                      // the getter
                      [](const offsets_pixel_t & pxl) {
                          return pxl.snr;
                      },
                      // the setter
                      [](offsets_pixel_t & pxl, float snr) {
                         pxl.snr = snr;
                      },
                      // the docstring
                      "the signal to noise ratio"
                      )
        // the covariance
        .def_property("covariance",
                      // the getter
                      [](const offsets_pixel_t & pxl) {
                          return pxl.covariance;
                      },
                      // the setter
                      [](offsets_pixel_t & pxl, float covariance) {
                         pxl.covariance = covariance;
                      },
                      // the docstring
                      "the covariance"
                      )
        // done
        ;

    // add the offset map layout interface
    pyOffsetsLayout
        .def_property_readonly("origin",
                               // the getter
                               &offsets_layout_t::origin,
                               // the docstring
                               "the origin of the offset map layout"
                               )

        .def_property_readonly("shape",
                               // the getter
                               &offsets_layout_t::shape,
                               // the docstring
                               "the shape of the offset map layout"
                               )

        // sizes of things: number of pixels
        .def_property_readonly("cells",
                               // the getter
                               &offsets_layout_t::cells,
                               // the docstring
                               "the number of cells in the offsets map"
                               )
        // and number of bytes
        .def_property_readonly("bytes",
                               // the getter
                               [](const offsets_layout_t & layout) {
                                   // easy enough
                                   return layout.cells() * sizeof(offsets_t::pixel_type);
                               },
                               // the docstring
                               "the memory footprint of the offsets map, in bytes"
                               )
        // done
        ;

    // add the Offsets index interface
    pyOffsetsIndex
        // access to individual ranks
        .def("__getitem__",
             // return the value of the requested rank
             [](const offsets_index_t & index, int rank) { return index[rank]; },
             // signature
             "rank"_a
             )
        // iteration support
        .def("__iter__",
             // make an iterator and return it
             [](const offsets_index_t & index) {
                 return py::make_iterator(index.begin(), index.end());
             },
             // make sure the index lives long enough
             py::keep_alive<0,1>()
             )
        // done
        ;

    // add the shape interface
    pyOffsetsShape
        // sizes of things: number of pixels
        .def_property_readonly("cells",
                               // the getter
                               &offsets_shape_t::cells,
                               // the docstring
                               "the number of pixels in the offset map"
                               )
        // access to individual ranks
        .def("__getitem__",
             // return the value of the requested rank
             [](const offsets_shape_t & shape, int idx) {
                 return shape[idx];
             },
             // signature
             "index"_a
             )
        // iteration support
        .def("__iter__",
             // make an iterator and return it
             [](const offsets_shape_t & shape) {
                 return py::make_iterator(shape.begin(), shape.end());
             },
             // make sure the shape lives long enough
             py::keep_alive<0,1>()
             )
        // done
        ;

    // the offset map interface
    pyOffsets
        // constructor
        .def(
             // the constructor wrapper
             py::init([](py::tuple pyShape) {
                          return offsets_constructor(pyShape);
                      }),
             // the signature
             "shape"_a
             )

        // accessors
        // sizes of things: number of cells
        .def_property_readonly("cells",
                      // the getter
                      &offsets_t::cells,
                      // the docstring
                      "the number of pixels in the offsets map"
                      )
        // sizes of things: memory footprint
        .def_property_readonly("bytes",
                      // the getter
                      &offsets_t::bytes,
                      // the docstring
                      "the amount of memory occupied by this offsets map, in bytes"
                      )
        // done
        ;

    // all done
    return;
}


// helper definitions
auto
ampcor::py::
offsets_constructor(py::tuple pyShape)
    -> offsets_t
{
    // extract the shape
    int rows = py::int_(pyShape[0]);
    int cols = py::int_(pyShape[1]);
    // make a shape
    offsets_t::layout_type::shape_type shape {rows, cols};
    // turn it into a layout
    offsets_t::layout_type layout { shape };
    // make a product specification out of the layout and return it
    return offsets_t { layout };
}


// end of file
