// -*- C++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// code guard
#if !defined(ampcor_libampcor_cuda_correlators_Sequential_h)
#define ampcor_libampcor_cuda_correlators_Sequential_h


// resource management and orchestration of the execution of the correlation plan
template <class slcT, class offsetsT>
class ampcor::cuda::correlators::Sequential {
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

    // the specification of the portion of the work assigned to me uses output tiles
    using offsets_layout_type = typename offsets_spec::layout_type;
    using offsets_shape_type = typename offsets_layout_type::shape_type;
    using offsets_index_type = typename offsets_layout_type::index_type;
    using offsets_layout_const_reference = const offsets_layout_type &;
    using offsets_shape_const_reference = const offsets_shape_type &;
    using offsets_index_const_reference = const offsets_index_type &;

    // temporary storage for detected tiles
    using arena_type = ampcor::dom::arena_raster_t<slc_value_type>;
    using const_arena_type = ampcor::dom::arena_const_raster_t<slc_value_type>;
    using arena_spec = typename arena_type::spec_type;
    using arena_value_type = typename arena_type::value_type;
    using arena_layout_type = typename arena_type::packing_type;
    using arena_index_type = typename arena_type::index_type;
    using arena_shape_type = typename arena_type::shape_type;
    // usage
    using arena_reference = arena_type &;
    using const_arena_const_reference = const const_arena_type &;
    using arena_layout_const_reference = typename arena_type::packing_const_reference;
    using arena_shape_const_reference = typename arena_type::shape_const_reference;

    // the tile plan
    using pairing_type = std::tuple<offsets_index_type, slc_index_type, slc_index_type>;
    using plan_type = std::vector<pairing_type>;
    using plan_const_reference = const plan_type &;

    // miscellaneous
    using string_type = string_t;
    using size_type = size_t;

    // device memory
    // some containers are raw pointers
    using dvector_type = devmem_t<slc_value_type>;
    using dvector_reference = dvector_type &;
    using dvector_const_reference = const dvector_type &;
    // we have arenas over real values
    using dev_arena_type = devarena_raster_t<slc_value_type>;
    // and arenas over complex values
    using dev_carena_type = devarena_raster_t<slc_pixel_type>;

    // usage
    using dev_arena_reference = dev_arena_type &;
    using dev_arena_const_reference = const dev_arena_type &;
    using dev_carena_reference = dev_carena_type &;
    using dev_carena_const_reference = const dev_carena_type &;

    // metamethods
public:
    // destructor
    inline ~Sequential();
    // constructor
    inline
    Sequential(int rank,
               slc_const_reference ref, slc_const_reference sec, offsets_reference map,
               slc_shape_const_reference chip, slc_shape_const_reference window,
               int refineFactor, int refineMargin, int zoomFactor);

    // interface
public:
    // execute the correlation plan and adjust the offset map
    void adjust(offsets_layout_const_reference);

    // implementation details: top level steps
private:
    auto coarseCorrelation(offsets_layout_const_reference) -> void;
    auto refinedCorrelation(offsets_layout_const_reference) -> void;

    // implementation details: methods
public:
    // build and validate a work plan
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

    // go through all the necessary steps to compute the correlation surface
    auto _gamma(string_type, arena_reference, arena_reference) -> dev_arena_type;
    // compute and store the locations of the maxima of the correlation surface
    auto _maxcor(plan_const_reference, dev_arena_reference, slc_value_type) -> void;

    // reduce the tiles in {arena} to zero mean, and compute their variances
    auto _referenceStatistics(dev_arena_reference) -> dvector_type;
    // create sum area tables for the secondary amplitude tiles
    auto _secondarySumAreaTables(dev_arena_reference) -> dev_arena_type;
    // construct an arena with the means and variances of all possible placements of the
    // reference chip in the secondary window
    auto _secondaryStatistics(arena_layout_const_reference, arena_layout_const_reference,
                              dev_arena_reference) -> dev_arena_type;
    // compute the correlation surface
    auto _correlate(dev_arena_const_reference, dvector_reference,
                    dev_arena_const_reference, dev_arena_const_reference) -> dev_arena_type;

    // build a complex arena on the device for staging tiles for de-ramping and refinement
    auto _createComplexArena(int, slc_index_const_reference, slc_shape_const_reference)
        -> dev_carena_type;
    // prime a complex device arena with raster data
    auto _primeComplexArena(plan_const_reference,
                            slc_const_reference, slc_shape_const_reference, bool,
                            dev_carena_reference) -> void;
    // deramp
    auto _deramp(dev_carena_reference, slc_shape_const_reference) -> void;
    // spread the spectrum: part of tile refinement between forward and reverse FFTs
    auto _spreadSpectrum(dev_carena_reference, slc_shape_const_reference) -> void;

    // data
private:
    // my rank
    int _rank;
    // the input rasters
    slc_const_reference _ref;
    slc_const_reference _sec;
    // the output raster
    offsets_reference _map;

    // the shape of the tiles
    const slc_shape_type _refShape;
    const slc_shape_type _secShape;

    // the correlation surface refinement parameters
    const int _refineFactor;
    const int _refineMargin;
    const int _zoomFactor;

    // disabled methods
private:
    // constructors
    Sequential(const Sequential &) = delete;
    Sequential(Sequential &&) = delete;
    Sequential & operator=(const Sequential &) = delete;
    Sequential & operator=(Sequential &&) = delete;
};


// the inline definitions
#define ampcor_cuda_correlators_Sequential_icc
#include "Sequential.icc"
#undef ampcor_cuda_correlators_Sequential_icc


// code guard
#endif

// end of file
