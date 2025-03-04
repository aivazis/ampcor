// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2025 all rights reserved

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

    // for the maps
    using point_t = index2d_t;
    using points_t = std::vector<point_t>;

    // plan
    using pairing_t = std::tuple<int, index2d_t, index2d_t>;
    using plan_t = std::vector<pairing_t>;
}


// make the potentially large containers opaque
PYBIND11_MAKE_OPAQUE(ampcor::py::plan_t);
PYBIND11_MAKE_OPAQUE(ampcor::py::points_t);


// assemble the {ampcor} package namespace
namespace ampcor::py {
    // bindings of opaque types
    void opaque(py::module &);
    // exceptions
    void exceptions(py::module &);

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

    // access to the correlators
    void sequential(py::module &);
    // and plans
    void plan(py::module &);

    // viz
    void viz(py::module &);
}


#endif

// end of file
