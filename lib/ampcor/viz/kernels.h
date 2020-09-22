// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_viz_kernels_h)
#define ampcor_viz_kernels_h


// the HSB to RGB conversion kernel
// see the wikipedia article at {https://en.wikipedia.org/wiki/HSL_and_HSV#HSB_to_RGB}
auto
ampcor::viz::
kernelHSBtoRGB(int n, double hue, double saturation, double brightness)
    -> color_t
{
    auto k = std::fmod((n + hue/60.), 6.);
    auto a = saturation * std::max(0., std::min(k, std::min(4.-k, 1.)));
    auto v = brightness * (1 - a);

    // scale up and return
    return 255*v;
}


// the HSL to RGB conversion kernel
// see the wikipedia article at {https://en.wikipedia.org/wiki/HSL_and_HSV#HSL_to_RGB}
auto
ampcor::viz::
kernelHSLtoRGB(int n, double hue, double saturation, double luminosity)
    -> color_t
{
    auto k = std::fmod((n + hue/30.), 12.);
    auto a = saturation * std::min(luminosity, 1.0-luminosity);
    auto v = luminosity - a * std::max(-1., std::min(k-3., std::min(9.-k, 1.)));

    // scale up and return
    return 255*v;
}


#endif

// end of file
