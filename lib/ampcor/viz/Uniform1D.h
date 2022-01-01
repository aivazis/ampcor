// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2022 all rights reserved

// code guard
#if !defined(ampcor_viz_Uniform1D_h)
#define ampcor_viz_Uniform1D_h


// a uniform 1D interpolator
template <class iteratorT>
class ampcor::viz::Uniform1D {
    // types
public:
    // my template parameter
    using source_type = iteratorT;
    // and the way i actually use it
    using source_reference = source_type &;

    // individual color values are one byte wide
    using color_type = color_t;
    // i generate {r,g,b} triplets
    using rgb_type = rgb_t;

    // metamethods
public:
    inline
    Uniform1D(source_reference data,
              // color map configuration
              int bins = 32,
              // color
              double hue = 0, double saturation = 0.5,
              // brightness range
              double minBrightness = 0, double maxBrightness = 1,
              // the data space
              double minData = 0, double maxData = 1);

    // interface: pretend to be an iterator
public:
    // map the current data value to a color
    inline auto operator*() const -> rgb_type;
    // get the next value from the source; only support the prefix form, if possible
    inline auto operator++() -> void;

    // implementation details: data
private:
    int    _bins;
    double _hue, _saturation;
    double _minBrightness, _maxBrightness, _scaleBrightness;
    double _minData, _maxData, _scaleData;
    // the data source
    source_reference _source;


    // default metamethods
public:
    // destructor
    ~Uniform1D() = default;

    // constructors
    Uniform1D(const Uniform1D &) = default;
    Uniform1D(Uniform1D &&) = default;
    Uniform1D & operator=(const Uniform1D &) = default;
    Uniform1D & operator=(Uniform1D &&) = default;
};


// get the inline definitions
#define ampcor_viz_Uniform1D_icc
#include "Uniform1D.icc"
#undef ampcor_viz_Uniform1D_icc


#endif

// end of file
