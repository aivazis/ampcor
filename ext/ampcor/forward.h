// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_py_forward_h)
#define ampcor_py_forward_h


// the {ampcor} namespace
namespace ampcor::py {
    // bindings of opaque types
    void opaque(py::module &);
    // exceptions
    void exceptions(py::module &);

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
