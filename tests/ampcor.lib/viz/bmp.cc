// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2025 all rights reserved


// STL
#include <vector>
#include <tuple>
#include <fstream>
// get the encoder
#include <ampcor/viz.h>


// type aliases
using bmp_t = ampcor::viz::bmp_t;


// generate a bitmap
int main(int argc, char *argv[]) {
    // pick a height
    int height = 255;
    // and a width
    int width = 255;

    // make a vector of {rgb} triplets
    std::vector<bmp_t::rgb_type> data;
    // give it enough room
    data.reserve(width * height);
    // go through it
    for (auto idx = 0; idx < width*height; ++idx) {
        // and place color values
        data.emplace(data.end(), idx%256, idx%256, idx%256);
    }

    // make a bitmap
    bmp_t bmp(height, width);

    // point to the beginning of the data
    auto start = data.begin();
    // encode
    auto img = bmp.encode(start);

    // open a file
    std::ofstream str("chip.bmp", std::ios::out | std::ios::binary);
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
