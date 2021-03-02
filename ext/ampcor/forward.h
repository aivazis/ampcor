// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2021 all rights reserved

// code guard
#if !defined(ampcor_py_forward_h)
#define ampcor_py_forward_h


// useful type aliases
namespace ampcor::py {
    // 2d layouts
    using layout2d_t = dom::layout_t<2>;
    using order2d_t = layout2d_t::order_type;
    using shape2d_t = layout2d_t::shape_type;
    using index2d_t = layout2d_t::index_type;
    // 3d layouts
    using layout3d_t = dom::layout_t<3>;
    using order3d_t = layout3d_t::order_type;
    using shape3d_t = layout3d_t::shape_type;
    using index3d_t = layout3d_t::index_type;
}


// assemble the {ampcor} package namespace
namespace ampcor::py {
    // bindings of opaque types
    void opaque(py::module &);
    // exceptions
    void exceptions(py::module &);

    // layouts
    // 2d
    void order2d(py::module &);
    void index2d(py::module &);
    void shape2d(py::module &);
    void layout2d(py::module &);
    // 3d
    void order3d(py::module &);
    void index3d(py::module &);
    void shape3d(py::module &);
    void layout3d(py::module &);

    // the raster layout
    void raster_layout(py::module &);

    // SLC
    void slc(py::module &);
    void slc_raster(py::module &);
    void slc_const_raster(py::module &);

    // offset maps
    void offsets(py::module &);
    void offsets_raster(py::module &);
    void offsets_const_raster(py::module &);

    // arenas
    void arena(py::module &);
    void arena_const_raster(py::module &);
    // the arena layout
    void arena_layout(py::module &);

    // access to the correlators
    void sequential(py::module &);

    // viz
    void viz(py::module &);
}


#endif

// end of file
