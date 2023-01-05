// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2023 all rights reserved


// STL
#include <cmath>
#include <fstream>
// support
#include <p2/grid.h>

// get the encoder
#include <ampcor/viz.h>


// type aliases
// the data
// a 2D grid
using pack_t = pyre::grid::canonical_t<2>;
// of doubles on the heap
using storage_t = pyre::memory::heap_t<double>;
// assemble
using grid_t = pyre::grid::grid_t<pack_t, storage_t>;

// the color map
using uni1d_t = ampcor::viz::uni1d_t<grid_t::const_iterator>;
// the bitmap
using bmp_t = ampcor::viz::bmp_t;


// generate a biitmap
int main(int argc, char *argv[]) {
    // pick a height
    int height = 512;
    // and a width
    int width = 512;

    // make the shape
    pack_t::shape_type shape { width, height };
    // shift the zero
    pack_t::index_type origin { -width/2, -height/2 };
    // build the layout
    pack_t packing { shape, origin };
    // make the grid
    grid_t data { packing, packing.cells() };

    // make a length scale
    auto scale = 0.1 * (width*width + height*height);
    // sample and populate
    for (auto idx : data.layout()) {
        // unpack the index
        auto [x,y] = idx;
        // compute the value and store it
        data[idx] =  std::exp( -(x*x + y*y)/scale );
    }

    // point to the beginning of the value grid
    auto start = data.cbegin();
    // make a color map
    uni1d_t cmap(start,
                 1 << 6,     // number of bins
                 0, 1.,      // hue, saturation
                 0, 1.,      // color space: (min, max) brightness
                 0, 1        // data space: (min, max) values
                 );

    // make a bitmap
    bmp_t bmp(height, width);
    // encode
    auto img = bmp.encode(cmap);

    // open a file
    std::ofstream str("uni1d.bmp", std::ios::out | std::ios::binary);
    // if we succeeded
    if (str.is_open()) {
        // ask for the stream size
        auto bytes = bmp.bytes();
        // write
        str.write(img, bytes);
    }

    // all done
    return 0;
}


// end of file
