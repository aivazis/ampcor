// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2020 all rights reserved


// externals
#include "external.h"
// namespace setup
#include "forward.h"


// grid + memory
#include <p2/grid.h>


// add bindings to SLC rasters
void
ampcor::py::
slc(py::module &m) {
    // type aliases
    // pixels
    using pixel_t = std::complex<float>;
    // layout of an SLC raster
    using packing_t = pyre::grid::canonical_t<2>;
    // SLC products are memory mapped
    using storage_t = pyre::memory::map_t<pixel_t>;
    // put it all together
    using slc_t = pyre::grid::grid_t<packing_t, storage_t>;

    // the SLC interface
    py::class_<slc_t>(m, "SLC")
        // the static interface
        // the size of a pixel in bytes
        .def_property_readonly_static("pixelSize",
                                      // the getter
                                      [] (py::object) -> size_t {
                                          return sizeof(slc_t::value_type);
                                      },
                                      // the docstring
                                      "the size of an SLC pixel"
                                      )
        // done
        ;

    // all done
    return;
}


// end of file
