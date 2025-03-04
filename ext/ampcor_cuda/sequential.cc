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
// libampcor_cuda
#include <ampcor_cuda/correlators.h>


// type aliases
namespace ampcor::cuda::py {
    // the rasters
    using slc_raster_t = ampcor::dom::slc_const_raster_t;
    using offsets_raster_t = ampcor::dom::offsets_raster_t;

    // usage
    using slc_const_reference = const slc_raster_t &;
    using offsets_reference = offsets_raster_t &;

    // the worker
    using sequential_t = ampcor::cuda::correlators::sequential_t<slc_raster_t, offsets_raster_t>;
    using sequential_reference = sequential_t &;

    // for 2d shapes and indices
    using tup2d = std::tuple<int, int>;
}


// add bindings to the sequential correlator
void
ampcor::cuda::py::sequential(py::module & m)
{
    // the SLC interface
    py::class_<sequential_t>(m, "Sequential")
        // constructor
        .def(    // the wrapper
            py::init([](
                         // the worker rank
                         int rank,
                         // the input rasters
                         slc_const_reference ref, slc_const_reference sec,
                         // the output
                         offsets_reference map,
                         // reference and secondary tile shapes
                         sequential_t::slc_shape_type chip, sequential_t::slc_shape_type window,
                         // refinement and zoom control
                         size_t refineFactor, size_t refineMargin, size_t zoomFactor) {
                // build a worker
                auto worker = new sequential_t(
                    rank, ref, sec, map, chip, window, refineFactor, refineMargin, zoomFactor);
                // wrap it and return it
                return std::unique_ptr<sequential_t>(worker);
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


// end of file
