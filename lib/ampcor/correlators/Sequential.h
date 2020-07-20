// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_correlators_Sequential_h)
#define ampcor_correlators_Sequential_h


// a worker that executes a correlation plan one tile at a time
template <class productT>
class ampcor::correlators::Sequential {
    // types
public:
    // my template parameter
    using product_type = productT;
    using product_const_reference = const product_type &;
    // the product spec
    using spec_type = typename product_type::spec_type;

    // the product pixel type
    using cell_type = typename spec_type::pixel_type;
    // and its underlying support
    using value_type = typename spec_type::value_type;

    // tile shape
    using shape_type = typename spec_type::shape_type;
    using shape_const_reference = const shape_type &;
    // tile layout
    using layout_type = typename spec_type::layout_type;
    using layout_const_reference = const layout_type &;
    // indices
    using index_type = typename spec_type::index_type;
    using index_const_reference = const index_type &;

    // the size of things
    using size_type = size_t;

    // metamethods
public:
    // destructor
    inline ~Sequential();

    // constructor
    inline
    Sequential(size_type pairs,
               layout_const_reference ref, layout_const_reference sec,
               size_type refineFactor, size_type refineMargin,
               size_type zoomFactor);

    // accessors
public:
    inline auto pairs() const -> size_type;

    inline auto coarseArena() const -> const cell_type *;
    inline auto coarseArenaCells() const -> size_type;
    inline auto coarseArenaBytes() const -> size_type;
    inline auto coarseArenaStride() const -> size_type;

    // interface
public:
    inline void fillCoarseArena(cell_type = 0) const;
    inline void addReferenceTile(size_type pid, product_const_reference ref);
    inline void addSecondaryTile(size_type pid, product_const_reference ref);

    // implementation details: data
private:
    // abbreviations: ref: reference tile, sec: secondary tile, cor: correlation matrix
    // my workload
    const size_type _pairs;
    // the correlation surface refinement parameters
    const size_type _refineFactor;
    const size_type _refineMargin;
    const size_type _zoomFactor;

    // the shapes of things
    // initially
    const layout_type _refLayout;
    const layout_type _secLayout;
    const layout_type _corLayout;
    // after refinement
    const layout_type _refRefinedLayout;
    const layout_type _secRefinedLayout;
    const layout_type _corRefinedLayout;
    // after zooming
    const layout_type _corZoomedLayout;

    // the number of cells
    // initially
    const size_type _refCells;
    const size_type _secCells;
    const size_type _corCells;
    // after refinement
    const size_type _refRefinedCells;
    const size_type _secRefinedCells;

    // the stride from one tile pair to the net
    const size_type _coarseStride;
    const size_type _refinedStride;

    // memory footprint, in bytes
    // initially
    const size_type _refBytes;
    const size_type _secBytes;
    const size_type _corBytes;
    // after refinement
    const size_type _refRefinedBytes;
    const size_type _secRefinedBytes;

    // scratch space capacity
    const size_type _coarseArenaCells;
    const size_type _refinedArenaCells;
    // scratch space memory footprint
    const size_type _coarseArenaBytes;
    const size_type _refinedArenaBytes;
    // scratch space
    cell_type * _coarseArena;
    cell_type * _refinedArena;

    // disabled metamethods
public:
    // constructors
    Sequential(const Sequential &) = delete;
    Sequential(Sequential &&) = delete;
    Sequential & operator=(const Sequential &) = delete;
    Sequential & operator=(Sequential &&) = delete;
};


// the inline definitions
#define ampcor_correlators_Sequential_icc
#include "Sequential.icc"
#undef ampcor_correlators_Sequential_icc


# endif

// end of file
