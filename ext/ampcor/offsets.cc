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
    // the product spec
    using offsets_t = ampcor::dom::offsets_t;
    // its pixel type
    using offsets_cell_t = offsets_t::pixel_type;
    // the pixel uses the {slc_t} index
    using slc_index_t = ampcor::dom::slc_t::layout_type::index_type;
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
    py::class_<offsets_cell_t> pyCell(pyOffsets, "Cell");

    // the pixel definition
    pyCell
        // the reference index
        .def_property("ref",
                      // the getter
                      [](const offsets_cell_t & pxl) {
                          return pxl.ref;
                      },
                      // the setter
                      [](offsets_cell_t & pxl, std::pair<float, float> ref) {
                          pxl.ref = ref;
                      },
                      // the docstring
                      "the reference pixel indices"
                      )
        // the shift
        .def_property("delta",
                      // the getter
                      [](const offsets_cell_t & pxl) {
                          return pxl.shift;
                      },
                      // the setter
                      [](offsets_cell_t & pxl, std::pair<float, float> delta) {
                          pxl.shift = delta;
                      },
                      // the docstring
                      "the offset to the matching pixel in the secondary raster"
                      )
        // the correlation
        .def_property("gamma",
                      // the getter
                      [](const offsets_cell_t & pxl) {
                          return pxl.gamma;
                      },
                      // the setter
                      [](offsets_cell_t & pxl, float gamma) {
                         pxl.gamma = gamma;
                      },
                      // the docstring
                      "the value of the correlation"
                      )
        // the level of confidence in the mapping
        .def_property("confidence",
                      // the getter
                      [](const offsets_cell_t & pxl) {
                          return pxl.confidence;
                      },
                      // the setter
                      [](offsets_cell_t & pxl, float confidence) {
                         pxl.confidence = confidence;
                      },
                      // the docstring
                      "the level of confidence in the mapping"
                      )
        // the signal to noise ration
        .def_property("snr",
                      // the getter
                      [](const offsets_cell_t & pxl) {
                          return pxl.snr;
                      },
                      // the setter
                      [](offsets_cell_t & pxl, float snr) {
                         pxl.snr = snr;
                      },
                      // the docstring
                      "the signal to noise ratio"
                      )
        // the covariance
        .def_property("covariance",
                      // the getter
                      [](const offsets_cell_t & pxl) {
                          return pxl.covariance;
                      },
                      // the setter
                      [](offsets_cell_t & pxl, float covariance) {
                         pxl.covariance = covariance;
                      },
                      // the docstring
                      "the covariance"
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
        // the size of a pixel
        .def_property_readonly("bytesPerCell",
                               // the getter
                               [](const offsets_t &) {
                                   // easy enough
                                   return sizeof(offsets_t::pixel_type);
                               },
                               // the docstring
                               "memory footprint of a map cell, in bytes"
                               )

        // access to the layout
        .def_property_readonly("layout",
                               // the getter
                               &offsets_t::layout,
                               // the docstring
                               "the layout of the map raster"
                               )

        .def_property_readonly("shape",
                               // the getter
                               [](const offsets_t & spec) {
                                   // easy enough
                                   return spec.layout().shape();
                               },
                               // the docstring
                               "the shape of the map raster"
                               )

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
    offsets_t::layout_type::shape_type shape { rows, cols };
    // use it to build a layout
    offsets_t::layout_type layout { shape };

    // make a product specification and return it
    return offsets_t { layout };
}


// end of file
