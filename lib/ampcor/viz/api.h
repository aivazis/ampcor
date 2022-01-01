// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2022 all rights reserved

// code guard
#if !defined(ampcor_viz_api_h)
#define ampcor_viz_api_h


// publicly visible types
namespace ampcor::viz {
    // the microsoft bmp stream generator
    using bmp_t = BMP;

    // the interpolators
    template <class sourceT>
    using phase1d_t = Phase1D<sourceT>;

    template <class sourceT>
    using uni1d_t = Uniform1D<sourceT>;

    template <class sourceT>
    using complex_t = Complex<sourceT>;

    // the SLC pixel detector
    using slc_detector_t = SLCDetector;
    // the phase calculator
    using slc_phaser_t = SLCPhaser;
}


#endif

// end of file
