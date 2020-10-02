// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_viz_Complex_h)
#define ampcor_viz_Complex_h


// map complex values to (hue, brightness)
template <class iteratorT>
class ampcor::viz::Complex {
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
    Complex(source_reference data,
              // color map configuration
              int bins = 32,
              // color
              double saturation = 0.5,
              // brightness range
              double minBrightness = 0, double maxBrightness = 1,
              // the data space
              double minAmplitude = 0, double maxAmplitude = 1);

    // interface: pretend to be an iterator
public:
    // map the current data value to a color
    inline auto operator*() const -> rgb_type;
    // get the next value from the source; only support the prefix form, if possible
    inline auto operator++() -> void;

    // implementation details: data
private:
    int    _bins;
    double _saturation;
    // hue mapping
    double _scaleHue;
    // brightness mapping
    double _minBrightness, _maxBrightness, _scaleBrightness;
    double _minAmplitude, _maxAmplitude, _scaleAmplitude;

    // the data source
    source_reference _source;

    // radian to degrees
    const double _deg = 180 / (4.*std::atan(1.));

    // default metamethods
public:
    // destructor
    ~Complex() = default;

    // constructors
    Complex(const Complex &) = default;
    Complex(Complex &&) = default;
    Complex & operator=(const Complex &) = default;
    Complex & operator=(Complex &&) = default;
};


// get the inline definitions
#define ampcor_viz_Complex_icc
#include "Complex.icc"
#undef ampcor_viz_Complex_icc


#endif

// end of file
