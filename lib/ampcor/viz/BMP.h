// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_viz_BMP_h)
#define ampcor_viz_BMP_h


// a grid with a payload that's a microsoft bitmap
class ampcor::viz::BMP {
    // types
public:
    // just to make sure we are all om the same page
    using byte_type = char;
    // individual color values are one byte wide
    using color_type = byte_type;
    // a pixel is a triplet of color values: <blue, green, red>
    // note that this is backwards from what you think but that's how the file is laid out...
    using rgb_type = std::tuple<color_type, color_type, color_type>;

    // the buffers i generated are shared pointers to character arrays
    using buffer_type = std::shared_ptr<const byte_type>;

    // metamethods
public:
    inline BMP(long height, long width);

    // interface
public:
    // the total size of the bitmap
    inline auto bytes() const -> int;

    template <class iteratorT>
    inline auto encode(iteratorT source) const -> buffer_type;

    // implementation details: data
private:
    long _width;
    long _height;
    long _padBytesPerLine;
    long _payloadSz;
    long _bitmapSz;

    // known at compile time
    static constexpr int _fileHeaderSz = 14;
    static constexpr int _infoHeaderSz = 40;
    static constexpr int _payloadOffset = _fileHeaderSz + _infoHeaderSz;
    static constexpr int _pixelSz = sizeof(rgb_type);

    // default metamethods
public:
    // destructor
    ~BMP() = default;

    // constructors
    BMP(const BMP &) = default;
    BMP(BMP &&) = default;
    BMP & operator=(const BMP &) = default;
    BMP & operator=(BMP &&) = default;
};


// get the inline definitions
#define ampcor_viz_BMP_icc
#include "BMP.icc"
#undef ampcor_viz_BMP_icc


#endif

// end of file
