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
    // the layout parts
    using arena_layout_t = dom::layout_t<3>;
    using arena_index_t = arena_layout_t::index_type;
    using arena_shape_t = arena_layout_t::shape_type;
}


// add bindings for the arena layout parts
void
ampcor::py::
arena_layout(py::module &m) {
    // the arena layout
    auto pyArenaLayout = py::class_<arena_layout_t>(m, "ArenaLayout");
    // its index
    auto pyArenaIndex = py::class_<arena_index_t>(m, "ArenaIndex");
    // and its shape
    auto pyArenaShape = py::class_<arena_shape_t>(m, "ArenaShape");

    // add the arena layout interface
    pyArenaLayout
        // constructor from a tuple
        .def(py::init([](const arena_shape_t & shape, const arena_index_t origin) {
            // make one and return it
            return arena_layout_t { shape, origin };
        }))

        // properties
        .def_property_readonly("origin",
                               // the getter
                               &arena_layout_t::origin,
                               // the docstring
                               "the origin of the arena layout"
                               )

        .def_property_readonly("shape",
                               // the getter
                               &arena_layout_t::shape,
                               // the docstring
                               "the shape of the arena layout"
                               )

        // sizes of things: number of pixels
        .def_property_readonly("cells",
                               // the getter
                               &arena_layout_t::cells,
                               // the docstring
                               "the number of pixels in the arena"
                               )
        // methods
        .def("box",
             // the handler
             &arena_layout_t::box,
             // the signature
             "origin"_a, "shape"_a,
             // the docstring
             "carve a portion of a layout"
             )

        // iteration support
        .def("__iter__",
             // make an iterator and return it
             [](const arena_layout_t & layout) {
                 return py::make_iterator(layout.begin(), layout.end());
             },
             // make sure the index lives long enough
             py::keep_alive<0,1>()
             )
        // done
        ;

    // add the arena index interface
    pyArenaIndex
        // constructor from a tuple
        .def(py::init([](py::tuple idx) {
                 // make one and return it
                 return arena_index_t { py::int_(idx[0]), py::int_(idx[1]), py::int_(idx[2])};
             }))
        // access to individual ranks
        .def("__getitem__",
             // return the value of the requested rank
             [](const arena_index_t & index, int rank) {
                 return index[rank];
             },
             // signature
             "rank"_a
             )
        // iteration support
        .def("__iter__",
             // make an iterator and return it
             [](const arena_index_t & index) {
                 return py::make_iterator(index.begin(), index.end());
             },
             // make sure the index lives long enough
             py::keep_alive<0,1>()
             )
        // done
        ;

    // add the arena shape interface
    pyArenaShape
        // constructor from a tuple
        .def(py::init([](py::tuple idx) {
            // make one and return it
            return arena_shape_t { py::int_(idx[0]), py::int_(idx[1]), py::int_(idx[2]) };
        }))
        // sizes of things: number of pixels
        .def_property_readonly("cells",
                               // the getter
                               &arena_shape_t::cells,
                               // the docstring
                               "the number of pixels in the arena"
                               )
        // access to individual ranks
        .def("__getitem__",
             // return the value of the requested rank
             [](const arena_shape_t & shape, int idx) {
                 return shape[idx];
             },
             // signature
             "index"_a
             )
        // iteration support
        .def("__iter__",
             // make an iterator and return it
             [](const arena_shape_t & shape) {
                 return py::make_iterator(shape.begin(), shape.end());
             },
             // make sure the shape lives long enough
             py::keep_alive<0,1>()
             )
        // done
        ;

    // all done
    return;
}


// end of file
