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

    // the worker
    using sequential_t = ampcor::correlators::sequential_t<slc_raster_t, offsets_raster_t>;
    using sequential_reference = sequential_t &;
}


// helpers
namespace ampcor::py {
    // the constructor
    static inline auto
    constructor(size_t, py::tuple, py::tuple, size_t, size_t, size_t)
        -> unique_pointer<sequential_t>;

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
}


// add bindings to the sequential correlator
void
ampcor::py::
sequential(py::module &m) {
    // the SLC interface
    py::class_<sequential_t>(m, "Sequential")
        // constructor
        .def(// the wrapper
             py::init([](size_t pairs, py::tuple ref, py::tuple sec,
                         size_t refineFactor, size_t refineMargin, size_t zoomFactor)
                      { return constructor(pairs, ref, sec,
                                           refineFactor, refineMargin, zoomFactor);}),
             // the signature
             "pairs"_a, "ref"_a, "sec"_a,
             "refineFactor"_a, "refineMargin"_a, "zoomFactor"_a
             )
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
        // execute the correlation plan and adjust the offset map
        .def("adjust",
             // the handler
             &sequential_t::adjust,
             // the signature
             "map"_a,
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
constructor(size_t pairs, py::tuple ref, py::tuple sec,
            size_t refineFactor, size_t refineMargin, size_t zoomFactor )
    -> unique_pointer<sequential_t>
{
    // extract the shape of the reference tile
    sequential_t::tile_grid_shape_type refShape { pairs, ref[0].cast<int>(), ref[1].cast<int>() };
    // and the shape of the secondary tile
    sequential_t::tile_grid_shape_type secShape { pairs, sec[0].cast<int>(), sec[1].cast<int>() };

    // build the layout of the reference tile
    sequential_t::tile_grid_layout_type refLayout { refShape };
    // and the layout of the secondary tile
    sequential_t::tile_grid_layout_type secLayout { secShape };

    // build the worker and return it
    return std::unique_ptr<sequential_t>(new sequential_t(pairs,
                                                          refLayout, secLayout,
                                                          refineFactor, refineMargin, zoomFactor));
}


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


// end of file
