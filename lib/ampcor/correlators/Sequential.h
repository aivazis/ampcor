// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_correlators_Sequential_h)
#define ampcor_correlators_Sequential_h


// a worker that executes a correlation plan one tile at a time
template <class slcT, class offsetsT>
class ampcor::correlators::Sequential {
    // types
public:
    // my template parameters
    using slc_type = slcT;
    using offsets_type = offsetsT;
    // typical uses
    using slc_const_reference = const slc_type &;
    using offsets_reference = offsets_type &;
    // the product specs
    using slc_spec = typename slc_type::spec_type;
    using offsets_spec = typename offsets_type::spec_type;

    // the slc pixel base type
    using slc_value_type = typename slc_spec::value_type;
    // the slc pixel type
    using slc_pixel_type = typename slc_spec::pixel_type;

    // my constructor depends on the layout of the incoming tiles
    using slc_layout_type = typename slc_spec::layout_type;
    using slc_layout_const_reference = const slc_layout_type &;

    // my record of the original pair collation order
    using pid_grid = size_t *;

    // tile storage
    using tile_grid_type = pyre::grid::grid_t<pyre::grid::canonical_t<3>,
                                              pyre::memory::map_t<float>>;
    using tile_grid_layout_type = tile_grid_type::packing_type;
    using tile_grid_index_type = tile_grid_type::index_type;
    using tile_grid_shape_type = tile_grid_type::shape_type;
    // usage
    using tile_grid_layout_const_reference = const tile_grid_layout_type &;

    // the size of things
    using size_type = size_t;

    // metamethods
public:
    // destructor
    inline ~Sequential();

    // constructor
    inline
    Sequential(size_type pairs,
               tile_grid_layout_const_reference ref, tile_grid_layout_const_reference sec,
               size_type refineFactor, size_type refineMargin,
               size_type zoomFactor);

    // accessors
public:
    inline auto pairs() const -> size_type;

    // interface
public:
    // transfer, detect, and store a pair of tiles and record the id of this pairing; note that
    // {tid} tracks the pairings that were assigned to me, while {pid} remembers the collation
    // number of this pairing in the original plan, which may have involved invalid tiles so
    // there may be gaps
    inline void addTilePair(size_type tid, size_type pid,
                            slc_const_reference ref, slc_const_reference sec
                            );

    // execute the correlation plan and adjust the offset map
    void adjust(offsets_reference);

    // implementation details: data
private:
    // my workload
    const size_type _pairs;
    // the correlation surface refinement parameters
    const size_type _refineFactor;
    const size_type _refineMargin;
    const size_type _zoomFactor;

    // scratch space
    pid_grid _pids;
    // the grid of reference tiles
    tile_grid_type _refCoarse;
    tile_grid_type _secCoarse;

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
