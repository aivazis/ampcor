// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_correlators_Sequential_h)
#define ampcor_correlators_Sequential_h


// a worker that executes a correlation plan one tile at a time
template <class inputT, class outputT>
class ampcor::correlators::Sequential {
    // types
public:
    // my template parameters
    using input_type = inputT;
    using input_const_reference = const input_type &;
    using output_type = outputT;
    using output_reference = output_type &;

    // the product spec
    using spec_type = typename input_type::spec_type;

    // the product pixel type
    using cell_type = typename spec_type::pixel_type;
    // and its underlying support
    using value_type = typename spec_type::value_type;
    // pointer to values
    using pointer = value_type *;
    // const pointer to values
    using const_pointer = const value_type *;

    // tile shape
    using shape_type = typename input_type::shape_type;
    using shape_const_reference = typename input_type::shape_const_reference;
    // tile layout
    using layout_type = typename spec_type::layout_type;
    using layout_const_reference = typename spec_type::layout_const_reference;
    // indices
    using index_type = typename input_type::index_type;
    using index_const_reference = typename input_type::index_const_reference;

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

    inline auto coarseArena() const -> const_pointer;
    inline auto coarseArenaCells() const -> size_type;
    inline auto coarseArenaBytes() const -> size_type;
    inline auto coarseArenaStride() const -> size_type;

    // interface
public:
    // record the id of a given pairing; note that {tid} tracks the pairings that were assigned
    // to me, while {pid} remembers the sequence number of this pairing in the original plan,
    // which may have involved invalid tiles so there may be gaps
    inline void addPair(size_type tid, size_type pid);
    // transfer and detect data from the incoming rasters
    inline void addReferenceTile(size_type tid, input_const_reference ref);
    inline void addSecondaryTile(size_type tid, input_const_reference sec);

    // initializer; useful for debugging
    inline void fillCoarseArena(value_type = 0) const;

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
    size_type * _pids;
    pointer _coarseArena;
    pointer _refinedArena;

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
