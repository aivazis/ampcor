// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2025 all rights reserved

// code guard
#if !defined(ampcor_viz_Phase1D_h)
#define ampcor_viz_Phase1D_h


// a phase interpolator
template <class iteratorT>
class ampcor::viz::Phase1D {
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
    Phase1D(source_reference data,
            // color map configuration
            int bins = 32,
            // color
            double saturation = 0.5, double brightness = 0);

    // interface: pretend to be an iterator
public:
    // map the current data value to a color
    inline auto operator*() const -> rgb_type;
    // get the next value from the source; only support the prefix form, if possible
    inline auto operator++() -> void;

    // implementation details: data
private:
    int    _bins;
    double _saturation, _brightness;
    // scaling the data and the hue
    double _scale;
    // the data source
    source_reference _source;


    // default metamethods
public:
    // destructor
    ~Phase1D() = default;

    // constructors
    Phase1D(const Phase1D &) = default;
    Phase1D(Phase1D &&) = default;
    Phase1D & operator=(const Phase1D &) = default;
    Phase1D & operator=(Phase1D &&) = default;
};


// get the inline definitions
#define ampcor_viz_Phase1D_icc
#include "Phase1D.icc"
#undef ampcor_viz_Phase1D_icc


#endif

// end of file
