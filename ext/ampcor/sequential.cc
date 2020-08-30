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
    static inline auto
    constructor(int rank,
                slc_const_reference, slc_const_reference, offsets_reference,
                py::tuple, py::tuple, size_t, size_t, size_t)
        -> unique_pointer<sequential_t>;

#if MGA
    // add a reference tile
    static inline auto
    addTilePair(sequential_t &,
                // the local and global pair collation orders
                size_t, size_t,
                // the reference tile
                const slc_raster_t &, const slc_raster_t::layout_type &,
                // the secondary tile
                const slc_raster_t &, const slc_raster_t::layout_type &)
        -> sequential_reference;
#endif
}


// add bindings to the sequential correlator
void
ampcor::py::
sequential(py::module &m) {
    // the SLC interface
    py::class_<sequential_t>(m, "Sequential")
        // constructor
        .def(// the wrapper
             py::init([](int rank,
                         slc_const_reference ref, slc_const_reference sec,
                         offsets_reference map,
                         py::tuple chip, py::tuple window,
                         size_t refineFactor, size_t refineMargin, size_t zoomFactor) {
                 return constructor(rank,
                                    ref, sec, map,
                                    chip, window,
                                    refineFactor, refineMargin, zoomFactor);
             }),
             // the signature
             "rank"_a,
             "reference"_a, "secondary"_a, "map"_a,
             "chip"_a, "window"_a,
             "refineFactor"_a, "refineMargin"_a, "zoomFactor"_a
             )
#if MGA
        // add a pair of tiles
        .def("addTilePair",
             // the handler
             addTilePair,
             // the signature
             "tid"_a, "pid"_a,
             "referenceRaster"_a, "referenceTile"_a, "secondaryRaster"_a, "secondaryTile"_a,
             // the docstring
             "detect and trasfer a reference tile to the coarse arena"
             )
#endif
        // execute the correlation plan and adjust the offset map
        .def("adjust",
             // the handler
             &sequential_t::adjust,
             // the signature
             "origin"_a, "shape"_a,
             // the docstring
             "execute the correlation plan and adjust the {offsets} map"
             )
        // done
        ;

    // all done
    return;
}


// helpers
// worker constructor
auto
ampcor::py::
constructor(int rank,
            slc_const_reference ref, slc_const_reference sec, offsets_reference map,
            py::tuple chip, py::tuple window,
            size_t refineFactor, size_t refineMargin, size_t zoomFactor )
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
    auto worker = new sequential_t(rank,
                                   ref, sec, map,
                                   refShape, secShape,
                                   refineFactor, refineMargin, zoomFactor);

    // build the worker and return it
    return std::unique_ptr<sequential_t>(worker);
}


#if MGA
// add a reference tile to the worker's coarse arena
auto
ampcor::py::
addTilePair(sequential_t & worker,
            size_t tid, size_t pid,
            const slc_raster_t & refRaster, const slc_raster_t::layout_type & refTile,
            const slc_raster_t & secRaster, const slc_raster_t::layout_type & secTile )
    -> sequential_reference
{
    // build the reference chip
    auto refChip = refRaster.tile(refTile.origin(), refTile.shape());
    // build the secondary chip
    auto secChip = secRaster.tile(secTile.origin(), secTile.shape());

    // make a channel
    pyre::journal::debug_t channel("ampcor.sequential.reference");
    // sign on
    channel
        << "addTilePair: tile " << tid  << ", originally " << pid << pyre::journal::newline
        << "  reference: " << pyre::journal::newline
        << "    origin: " << refChip.layout().origin() << pyre::journal::newline
        << "    shape: " << refChip.layout().shape() << pyre::journal::newline
        << "    data: " << refChip.data().get() << pyre::journal::newline
        << "  secondary: " << pyre::journal::newline
        << "    origin: " << secChip.layout().origin() << pyre::journal::newline
        << "    shape: " << secChip.layout().shape() << pyre::journal::newline
        << "    data: " << secChip.data().get() << pyre::journal::newline
        << pyre::journal::endl(__HERE__);

    // engage
    worker.addTilePair(tid, pid, refChip, secChip);
    // all done
    return worker;
}
#endif


// end of file
