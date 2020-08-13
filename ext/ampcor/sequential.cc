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
    using sequential_t = ampcor::correlators::sequential_t<slc_raster_t>;
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
    addReferenceTile(sequential_t &, const slc_raster_t &,
                     size_t, const slc_raster_t::layout_type &) -> sequential_reference;
    // add a secondary tile
    static inline auto
    addSecondaryTile(sequential_t &, const slc_raster_t &,
                     size_t, const slc_raster_t::layout_type &) -> sequential_reference;
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
        // record the original collation number of this pairing
        .def("addPair",
             &sequential_t::addPair,
             // the signature
             "tid"_a, "pid"_a,
             // the docstring
             "record the collation number of this pair"
             )
        // add a reference tile
        .def("addReferenceTile",
             addReferenceTile,
             // the signature
             "raster"_a, "tid"_a, "tile"_a,
             // the docstring
             "detect and trasfer a reference tile to the coarse arena"
             )
        // add a secondary tile
        .def("addSecondaryTile",
             addSecondaryTile,
             // the signature
             "raster"_a, "tid"_a, "tile"_a,
             // the docstring
             "detect and trasfer a secondary tile to the coarse arena"
             )
        // execute the correlation plan and adjust the offset map
        .def("adjust",
             [](sequential_t & worker, offsets_raster_t & map) {
                 // all done
                 return;
             },
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
    slc_raster_t::shape_type refShape { ref[0].cast<int>(), ref[1].cast<int>() };
    // and the shape of the secondary tile
    slc_raster_t::shape_type secShape { sec[0].cast<int>(), sec[1].cast<int>() };

    // build the layout of the reference tile
    slc_raster_t::packing_type refLayout { refShape };
    // and the layout of the secondary tile
    slc_raster_t::packing_type secLayout { secShape };

    // build the worker and return it
    return std::unique_ptr<sequential_t>(new sequential_t(pairs,
                                                          refLayout, secLayout,
                                                          refineFactor, refineMargin, zoomFactor));
}


// add a reference tile to the worker's coarse arena
auto
ampcor::py::
addReferenceTile(sequential_t & worker,
                 const slc_raster_t & raster,
                 size_t tid, const slc_raster_t::layout_type & chip) -> sequential_reference
{
    // make the tile
    auto tile = raster.tile(chip.origin(), chip.shape());

    // make a channel
    pyre::journal::debug_t channel("ampcor.sequential.reference");
    // sign on
    channel
        << "addReferenceTile: tile #" << tid << pyre::journal::newline
        << "  raster:" << pyre::journal::newline
        << "    shape: " << raster.layout().shape() << pyre::journal::newline
        << "    data: " << raster.data().get() << pyre::journal::newline
        << "  spec: " << pyre::journal::newline
        << "    origin: " << chip.origin() << pyre::journal::newline
        << "    shape: " << chip.shape() << pyre::journal::newline
        << "  tile: " << pyre::journal::newline
        << "    origin: " << tile.layout().origin() << pyre::journal::newline
        << "    shape: " << tile.layout().shape() << pyre::journal::newline
        << "    data: " << tile.data().get() << pyre::journal::newline
        << pyre::journal::endl(__HERE__);

    // engage
    worker.addReferenceTile(tid, tile);

    // all done
    return worker;
}


// add a secondary tile to the worker's coarse arena
auto
ampcor::py::
addSecondaryTile(sequential_t & worker,
                 const slc_raster_t & raster,
                 size_t tid, const slc_raster_t::layout_type & chip) -> sequential_reference
{
    // make the tile
    auto tile = raster.tile(chip.origin(), chip.shape());

    // make a channel
    pyre::journal::debug_t channel("ampcor.sequential.reference");
    // sign on
    channel
        << "addSeoncaryTile: tile #" << tid << pyre::journal::newline
        << "  raster:" << pyre::journal::newline
        << "    shape: " << raster.layout().shape() << pyre::journal::newline
        << "    data: " << raster.data().get() << pyre::journal::newline
        << "  spec: " << pyre::journal::newline
        << "    origin: " << chip.origin() << pyre::journal::newline
        << "    shape: " << chip.shape() << pyre::journal::newline
        << "  tile: " << pyre::journal::newline
        << "    origin: " << tile.layout().origin() << pyre::journal::newline
        << "    shape: " << tile.layout().shape() << pyre::journal::newline
        << "    data: " << tile.data().get() << pyre::journal::newline
        << pyre::journal::endl(__HERE__);

    // engage
    worker.addSecondaryTile(tid, tile);

    // all done
    return worker;
}


// end of file
