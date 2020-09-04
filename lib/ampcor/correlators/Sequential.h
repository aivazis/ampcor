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

    // my constructor and implementation depend on the layout of input tiles
    using slc_layout_type = typename slc_spec::layout_type;
    using slc_shape_type = typename slc_layout_type::shape_type;
    using slc_index_type = typename slc_layout_type::index_type;
    using slc_layout_const_reference = const slc_layout_type &;
    using slc_shape_const_reference = const slc_shape_type &;
    using slc_index_const_reference = const slc_index_type &;

    // the specification of the portion of the work assigned to me uses output tiles`
    using offsets_layout_type = typename offsets_spec::layout_type;
    using offsets_shape_type = typename offsets_layout_type::shape_type;
    using offsets_index_type = typename offsets_layout_type::index_type;
    using offsets_layout_const_reference = const offsets_layout_type &;
    using offsets_shape_const_reference = const offsets_shape_type &;
    using offsets_index_const_reference = const offsets_index_type &;

    // temporary storage for detected tiles
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

    // temporary storage for complex tiles
    using carena_type = ampcor::dom::arena_raster_t<slc_pixel_type>;
    using const_carena_type = ampcor::dom::arena_const_raster_t<slc_pixel_type>;
    using carena_reference = carena_type &;
    using carena_spec = typename carena_type::spec_type;
    using carena_value_type = typename carena_type::value_type;
    using carena_layout_type = typename carena_type::packing_type;
    using carena_index_type = typename carena_type::index_type;
    using carena_shape_type = typename carena_type::shape_type;
    // usage
    using const_carena_const_reference = const const_carena_type &;
    using carena_layout_const_reference = typename carena_type::packing_const_reference;
    using carena_index_const_reference = typename carena_type::index_const_reference;
    using carena_shape_const_reference = typename carena_type::shape_const_reference;

    // 1d vectors
    using vector_type = std::valarray<arena_value_type>;
    using vector_pointer = std::shared_ptr<vector_type>;
    // the tile plan
    using pairing_type = std::tuple<offsets_index_type, slc_index_type, slc_index_type>;
    using plan_type = std::vector<pairing_type>;
    using plan_const_reference = const plan_type &;

    // miscellaneous
    using string_type = string_t;

    // non-trivial metamethods
public:
    // constructor
    inline
    Sequential(int rank,
               slc_const_reference ref, slc_const_reference sec, offsets_reference map,
               slc_shape_const_reference refShape, slc_shape_const_reference secShape,
               int refineFactor, int refineMargin, int zoomFactor);

    // interface
public:
    // execute the correlation plan and adjust the offset map
    void adjust(offsets_layout_const_reference);

    // implementation details: top level steps
public:
    void coarseCorrelation(offsets_layout_const_reference);
    void refinedCorrelation(offsets_layout_const_reference);

    // implementation details: methods
public:
    auto _assemblePlan(offsets_layout_const_reference,
                       slc_shape_const_reference, slc_shape_const_reference) -> plan_type;
    // create a tile arena
    auto _createAmplitudeArena(string_type name, int pairs,
                               slc_shape_const_reference tileShape,
                               slc_index_const_reference tileOrigin)
        -> arena_type;
    // detect and transfer reference and secondary tiles into their respective arenas
    auto _detect(plan_const_reference plan, arena_reference refArena, arena_reference secArena)
        -> void;
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
    auto _maxcor(plan_const_reference, offsets_reference, const_arena_const_reference);
    // build complex arenas for storing tiles for deramping and refininment
    auto _createComplexArena(string_type, int, slc_shape_const_reference) -> carena_type;
    // fill the complex arenas with pixels from the rasters
    auto _primeComplexArenas(plan_const_reference, carena_reference, carena_reference) -> void;
    // deramp
    auto _deramp(carena_reference) -> void;
    // refine
    auto _refine(carena_reference) -> void;
    // spectrum spread
    auto _spreadSpectrum(carena_reference arena, int factor) -> void;

    // implementation details: data
private:
    // my rank; useful in parallel mode
    int _rank;
    // the input rasters
    slc_const_reference _ref;
    slc_const_reference _sec;
    offsets_reference _map;
    // the shapes of tiles
    slc_shape_type _refShape;
    slc_shape_type _secShape;
    // the correlation surface refinement parameters
    const int _refineFactor;
    const int _refineMargin;
    const int _zoomFactor;

    // default metamethods
public:
    ~Sequential() = default;

    // disabled metamethods
private:
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
