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
#include <ampcor/correlators.h>


// type aliases
namespace ampcor::py {
    // the rasters
    using slc_raster_t = ampcor::dom::slc_const_raster_t;
    using offsets_raster_t = ampcor::dom::offsets_raster_t;

    // usage
    using slc_const_reference = const slc_raster_t &;
    using offsets_reference = offsets_raster_t &;

    // the worker
    using sequential_t = ampcor::correlators::sequential_t<slc_raster_t, offsets_raster_t>;
    using sequential_reference = sequential_t &;
}


// helpers
namespace ampcor::py {
    // the constructor
    static inline auto constructor(
        int rank, slc_const_reference, slc_const_reference, offsets_reference, py::tuple, py::tuple,
        size_t, size_t, size_t) -> unique_pointer<sequential_t>;
}


// add bindings to the sequential correlator
void
ampcor::py::sequential(py::module & m)
{
    // the SLC interface
    py::class_<sequential_t>(m, "Sequential")
        // constructor
        .def(    // the wrapper
            py::init([](int rank, slc_const_reference ref, slc_const_reference sec,
                        offsets_reference map, py::tuple chip, py::tuple window,
                        size_t refineFactor, size_t refineMargin, size_t zoomFactor) {
                return constructor(
                    rank, ref, sec, map, chip, window, refineFactor, refineMargin, zoomFactor);
            }),
            // the signature
            "rank"_a, "reference"_a, "secondary"_a, "map"_a, "chip"_a, "window"_a, "refineFactor"_a,
            "refineMargin"_a, "zoomFactor"_a)

        // execute the correlation plan and adjust the offset map
        .def(
            "adjust",
            // the handler
            &sequential_t::adjust,
            // the signature
            "box"_a,
            // the docstring
            "execute the correlation plan and adjust the {offsets} map")
        // done
        ;

    // all done
    return;
}


// helpers
// worker constructor
auto
ampcor::py::constructor(
    int rank, slc_const_reference ref, slc_const_reference sec, offsets_reference map,
    py::tuple chip, py::tuple window, size_t refineFactor, size_t refineMargin, size_t zoomFactor)
    -> unique_pointer<sequential_t>
{
    // unpack the chip
    size_t chip_0 = py::int_(chip[0]);
    size_t chip_1 = py::int_(chip[1]);
    // unpack the padding
    size_t win_0 = py::int_(window[0]);
    size_t win_1 = py::int_(window[1]);

    // build the shape of the reference tiles
    sequential_t::slc_shape_type refShape { chip_0, chip_1 };
    // build the shape of the secondary tiles
    sequential_t::slc_shape_type secShape { win_0, win_1 };

    // build a worker
    auto worker = new sequential_t(
        rank, ref, sec, map, refShape, secShape, refineFactor, refineMargin, zoomFactor);

    // build the worker and return it
    return std::unique_ptr<sequential_t>(worker);
}


// end of file
