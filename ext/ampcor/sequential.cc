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
using slc_t = ampcor::dom::slc_const_raster_t;
using sequential_t = ampcor::correlators::sequential_t<slc_t>;
using sequential_reference = sequential_t &;


// helpers
namespace ampcor::py {
    // the constructor
    static inline auto
    constructor(size_t, py::tuple, py::tuple, size_t, size_t, size_t)
        -> unique_pointer<sequential_t>;

    // add a reference tile
    static inline auto
    addReferenceTile(sequential_t &, const slc_t &,
                     size_t, py::tuple, py::tuple) -> sequential_reference;
    // add a secondary tile
    static inline auto
    addSecondaryTile(sequential_t &, const slc_t &,
                     size_t, py::tuple, py::tuple) -> sequential_reference;
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
        // add a reference tile
        .def("addReferenceTile",
             addReferenceTile,
             "raster"_a, "tid"_a, "origin"_a, "shape"_a
             )
        // add a secondary tile
        .def("addSecondaryTile",
             addSecondaryTile,
             "raster"_a, "tid"_a, "origin"_a, "shape"_a
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
    slc_t::shape_type refShape { ref[0].cast<int>(), ref[1].cast<int>() };
    // and the shape of the secondary tile
    slc_t::shape_type secShape { sec[0].cast<int>(), sec[1].cast<int>() };

    // build the layout of the reference tile
    slc_t::packing_type refLayout { refShape };
    // and the layout of the secondary tile
    slc_t::packing_type secLayout { secShape };

    // build the worker and return it
    return std::unique_ptr<sequential_t>(new sequential_t(pairs,
                                                          refLayout, secLayout,
                                                          refineFactor, refineMargin, zoomFactor));
}


// add a reference tile to the worker's coarse arena
auto
ampcor::py::
addReferenceTile(sequential_t & worker,
                 const slc_t & raster,
                 size_t tid, py::tuple pyOrigin, py::tuple pyShape) -> sequential_reference
{
    // alias for the index
    using idx_t = slc_t::index_type;
    // and the type of its ranks
    using idx_rank_t = idx_t::value_type;
    // make an index out of the origin
    slc_t::index_type origin { pyOrigin[0].cast<idx_rank_t>(), pyOrigin[1].cast<idx_rank_t>() };

    // alias for the shape
    using shp_t = slc_t::shape_type;
    // and the type of its ranks
    using shp_rank_t = shp_t::value_type;
    // make a shape out of the extent
    slc_t::shape_type shape { pyShape[0].cast<shp_rank_t>(), pyShape[1].cast<shp_rank_t>() };

    // make the tile
    auto tile = raster.tile(origin, shape);

    // make a channel
    pyre::journal::debug_t channel("ampcor.sequential.reference");
    // sign on
    channel
        << "addReferenceTile: tile #" << tid << pyre::journal::newline
        << "  raster:" << pyre::journal::newline
        << "    shape: " << raster.layout().shape() << pyre::journal::newline
        << "    data: " << raster.data().get() << pyre::journal::newline
        << "  spec: " << pyre::journal::newline
        << "    origin: " << origin << pyre::journal::newline
        << "    shape: " << shape << pyre::journal::newline
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
                 const slc_t & raster,
                 size_t tid, py::tuple pyOrigin, py::tuple pyShape) -> sequential_reference
{
    // alias for the index
    using idx_t = slc_t::index_type;
    // and the type of its ranks
    using idx_rank_t = idx_t::value_type;
    // make an index out of the origin
    slc_t::index_type origin { pyOrigin[0].cast<idx_rank_t>(), pyOrigin[1].cast<idx_rank_t>() };

    // alias for the shape
    using shp_t = slc_t::shape_type;
    // and the type of its ranks
    using shp_rank_t = shp_t::value_type;
    // make a shape out of the extent
    slc_t::shape_type shape { pyShape[0].cast<shp_rank_t>(), pyShape[1].cast<shp_rank_t>() };

    // make the tile
    auto tile = raster.tile(origin, shape);

    // make a channel
    pyre::journal::debug_t channel("ampcor.sequential.reference");
    // sign on
    channel
        << "addSeoncaryTile: tile #" << tid << pyre::journal::newline
        << "  raster:" << pyre::journal::newline
        << "    shape: " << raster.layout().shape() << pyre::journal::newline
        << "    data: " << raster.data().get() << pyre::journal::newline
        << "  spec: " << pyre::journal::newline
        << "    origin: " << origin << pyre::journal::newline
        << "    shape: " << shape << pyre::journal::newline
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
