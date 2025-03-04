// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2025 all rights reserved

// code guard
#if !defined(ampcor_viz_BMP_h)
#define ampcor_viz_BMP_h


// a microsoft bitmap generator
class ampcor::viz::BMP {
    // types
public:
    // just to make sure we are all on the same page
    using byte_type = char;
    // individual color values are one byte wide
    using color_type = byte_type;
    // a pixel is a triplet of color values: <blue, green, red>
    // note that this is backwards from what you think but that's how the file is laid out...
    using rgb_type = std::tuple<color_type, color_type, color_type>;

    // the buffers i generated are shared pointers to character arrays
    using buffer_type = byte_type *;

    // metamethods
public:
    // destructor
    inline ~BMP();
    // constructor
    inline BMP(int height, int width);
    // move semantics
    inline BMP(BMP &&);
    inline BMP & operator=(BMP &&);

    // accessors
public:
    // the total size of the bitmap
    inline auto bytes() const -> int;
    inline auto data() const -> buffer_type;

    // interface
public:
    // read data from {source} and encode it in the bitmap
    template <class iteratorT>
    inline auto encode(iteratorT & source, bool topdown = true) const -> buffer_type;

    // implementation details: data
private:
    int _width;
    int _height;
    int _padBytesPerLine;
    int _payloadSz;
    int _bitmapSz;
    // my data buffer
    byte_type * _data;

    // known at compile time
    static constexpr int _fileHeaderSz = 14;
    static constexpr int _infoHeaderSz = 40;
    static constexpr int _bitPlanes = 24;
    static constexpr int _payloadOffset = _fileHeaderSz + _infoHeaderSz;
    static constexpr int _pixelSz = sizeof(rgb_type);

    // disabled metamethods
private:
    // constructors
    BMP(const BMP &) = delete;
    BMP & operator=(const BMP &) = delete;
};


// get the inline definitions
#define ampcor_viz_BMP_icc
#include "BMP.icc"
#undef ampcor_viz_BMP_icc


#endif

// end of file
