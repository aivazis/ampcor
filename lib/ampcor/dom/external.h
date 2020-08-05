// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_dom_external_h)
#define ampcor_dom_external_h


// STL
#include <complex>

// externals
#include <p2/grid.h>


// type aliases
namespace ampcor::dom {
    // sizes of things
    using size_t = std::size_t;

    // complex numbers
    template <typename T>
    using complex_t = std::complex<T>;

    // storage strategies
    // memory mapped data
    template <typename pixelT, bool isReadOnly = true>
    using mmap_t = std::conditional_t<isReadOnly,
                                      pyre::memory::constmap_t<pixelT>,
                                      pyre::memory::map_t<pixelT>
                                      >;

    // most products are {lines}x{samples} packed in row-major order
    template <size_t N>
    using layout_t = pyre::grid::canonical_t<N>;

    // pull {pyre::grid:grid_t}, with a twist
    template <class specT, template <typename, bool> class storageT, bool isReadOnly = true>
    using grid_t = pyre::grid::grid_t<typename specT::layout_type,
                                      storageT<typename specT::pixel_type, isReadOnly>>;
}

#endif

// end of file
