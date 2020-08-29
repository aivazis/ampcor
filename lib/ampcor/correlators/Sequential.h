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
    using pid_grid = int *;

    // temporary tile storage
    using arena_type = ampcor::dom::arena_raster_t<slc_value_type>;
    using const_arena_type = ampcor::dom::arena_const_raster_t<slc_value_type>;
    using arena_reference = arena_type &;
    using arena_spec = typename arena_type::spec_type;
    using arena_value_type = typename arena_type::value_type;
    using arena_layout_type = typename arena_type::packing_type;
    using arena_index_type = typename arena_type::index_type;
    using arena_shape_type = typename arena_type::shape_type;
    // usage
    using const_arena_const_reference = const const_arena_type &;
    using arena_layout_const_reference = typename arena_type::packing_const_reference;
    using arena_shape_const_reference = typename arena_type::shape_const_reference;

    // 1d vectors
    using vector_type = std::valarray<arena_value_type>;
    using vector_pointer = std::shared_ptr<vector_type>;

    // miscellaneous
    using string_type = string_t;

    // metamethods
public:
    // destructor
    inline ~Sequential();

    // constructor
    inline
    Sequential(slc_const_reference ref, slc_const_reference sec, offsets_reference map,
               arena_layout_const_reference refLayout, arena_layout_const_reference secLayout,
               int refineFactor, int refineMargin,
               int zoomFactor);

    // accessors
public:
    inline auto pairs() const -> int;

    // interface
public:
    // transfer, detect, and store a pair of tiles and record the id of this pairing; note that
    // {tid} tracks the pairings that were assigned to me, while {pid} remembers the collation
    // number of this pairing in the original plan, which may have involved invalid tiles so
    // there may be gaps
    inline void addTilePair(int tid, int pid,
                            slc_const_reference ref, slc_const_reference sec);

    // execute the correlation plan and adjust the offset map
    auto adjust();

    // implementation details: methods
public:
    // reduce the tiles in {arena} to zero mean, and compute their variances
    auto _referenceStatistics(arena_reference) -> vector_pointer;

    // build sum tables for the tiles in {arena}
    auto _secondarySumAreaTables(string_type, arena_reference) -> const_arena_type;
    // construct an arena with the means and variances of all possible placements of the
    // reference chip in the secondary window
    auto _secondaryStatistics(string_type,
                              arena_layout_const_reference, arena_layout_const_reference,
                              const_arena_const_reference) -> const_arena_type;
    // compute the correlation surface
    auto _correlate(string_type,
                    const_arena_const_reference, vector_pointer,
                    const_arena_const_reference, const_arena_const_reference) -> const_arena_type;
    // compute and store the locations of the maxima of the correlation surface
    auto _maxcor(const_arena_const_reference, offsets_reference);

    // implementation details: data
private:
    // the input rasters
    slc_const_reference _ref;
    slc_const_reference _sec;
    offsets_reference _map;
    // my workload
    const int _pairs;
    // the correlation surface refinement parameters
    const int _refineFactor;
    const int _refineMargin;
    const int _zoomFactor;

    // scratch space
    pid_grid _pids;
    // the grid of reference tiles
    arena_type _refCoarse;
    arena_type _secCoarse;

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
