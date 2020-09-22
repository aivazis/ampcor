// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_viz_forward_h)
#define ampcor_viz_forward_h


// set up the namespace
namespace ampcor::viz {
    // just to make sure we are all on the same page
    using byte_t = char;
    // individual color values are one byte wide
    using color_t = byte_t;
    // i generate {r,g,b} triplets
    using rgb_t = std::tuple<color_t, color_t, color_t>;

    // microsoft bmp
    class BMP;

    // one dimensional linear interpolator
    template <class iteratorT> class Uniform1D;
}


// useful functions
namespace ampcor::viz {
    // the HSB to RGB conversion kernel
    // see the wikipedia article at {https://en.wikipedia.org/wiki/HSL_and_HSV#HSB_to_RGB}
    inline
    auto kernelHSBtoRGB(int n, double hue, double saturation, double brightness) -> color_t;

    // the HSL to RGB conversion kernel
    // see the wikipedia article at {https://en.wikipedia.org/wiki/HSL_and_HSV#HSL_to_RGB}
    inline
    auto kernelHSLtoRGB(int n, double hue, double saturation, double luminosity) -> color_t;
}


#endif

// end of file
