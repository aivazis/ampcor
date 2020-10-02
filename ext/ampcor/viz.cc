// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2020 all rights reserved


// externals
#include "external.h"
// namespace setup
#include "forward.h"
// libampcor
#include <ampcor/dom.h>
#include <ampcor/viz.h>

// type aliases
namespace ampcor::py {
    // {viz} only cares about {const_raster_t} types
    // arenas of floats
    using arena_t = ampcor::dom::arena_const_raster_t<float>;
    // complex arenas
    using carena_t = ampcor::dom::arena_const_raster_t<std::complex<float>>;
    // uses
    using arena_const_reference = const arena_t &;

    // slc
    using slc_t = ampcor::dom::slc_const_raster_t;
    // uses
    using slc_const_reference = const slc_t &;

    // bitmap
    using bmp_t = ampcor::viz::bmp_t;

    // slc color maps
    using slc_uni1d_t = ampcor::viz::uni1d_t<ampcor::viz::slc_detector_t>;
    using slc_phase1d_t = ampcor::viz::phase1d_t<ampcor::viz::slc_phaser_t>;
    using slc_complex_t = ampcor::viz::complex_t<slc_t::const_iterator>;

    // arena color map
    using arena_uni1d_t = ampcor::viz::uni1d_t<arena_t::const_iterator>;
}


// visualization support
void
ampcor::py::
viz(py::module & m) {
    // put all this in the {viz} namespace
    auto pyviz = m.def_submodule("viz", "Support for visualization");

    // publish BMP; it supports the {buffer} protocol
    py::class_<bmp_t>(pyviz, "BMP", py::buffer_protocol())
        // access the underlying data
        .def_buffer([](bmp_t & bmp) -> py::buffer_info {
            // build one and return it
            return py::buffer_info (bmp.data(),
                                    sizeof(bmp_t::byte_type),
                                    py::format_descriptor<bmp_t::byte_type>::format(),
                                    1,
                                    { bmp.bytes() },
                                    { 1 });
        })
        // done
        ;

    // generate a bitmap for a specific real arena tile
    pyviz.def("arenaTile",
          // the function
          [](arena_const_reference arena, int tid) -> bmp_t {
              // make a channel
              pyre::journal::info_t channel("ampcor.viz.arena.tile");

              // isolate the portion of the arena that i care about
              auto tile = arena.box(arena.spec().tile(tid));
              // get its layout
              auto layout = tile.layout();
              // origin and shape
              auto origin = layout.origin();
              auto shape = layout.shape();

              // pick the number of bins
              int bins = 1 << 5;
              // find the {min} and {max} elements
              auto min = tile[origin];
              auto max = tile[origin];
              // by going through all the tile entries
              for (auto v : tile) {
                  // and checking whether this value is larger than the current maximum
                  if (v > max) {
                      // in which case, just replace it
                      max = v;
                  }
                  // and whether it's smaller than the current minimum
                  if (v < min) {
                      // in which case, just replace it
                      min = v;
                  }
              }
              // pick hue and saturation
              auto hue = 0;
              auto saturation = 0;
              // pick a brightness range
              auto minB = 0.2;
              auto maxB = 1.0;

              // show me
              channel
                  << "tile:" << pyre::journal::newline
                  << "  origin: " << origin << pyre::journal::newline
                  << "  shape: " << shape << pyre::journal::newline
                  << "  viz:" << pyre::journal::newline
                  << "    bins: " << bins << pyre::journal::newline
                  << "    hue: " << hue << pyre::journal::newline
                  << "    saturation: " << saturation << pyre::journal::newline
                  << "    data: [ " << min << ", " << max << "]" << pyre::journal::newline
                  << "    brightness: [ " << minB << ", " << maxB << "]" << pyre::journal::newline
                  << pyre::journal::endl;

              // point to the beginning of the data
              auto start = tile.cbegin();

              // make the color map
              arena_uni1d_t cmap(start,
                                 bins,
                                 hue, saturation,
                                 minB,maxB,
                                 min, max);

              // make a bitmap object
              bmp_t bmp(shape[1], shape[2]);
              // encode the data using the color map
              bmp.encode(cmap);

              // and return it
              return bmp;
          },
          // the signature
          "arena"_a, "tileid"_a,
          // the docstring
          "generate a BMP for the {arena} tile at the given {tileid}"
          )

        // generate a bitmap out of the amplitude of an SLC tile
        .def("slc",
             // the handler
             [](slc_const_reference slc,
                std::pair<int, int> tileOrigin, std::pair<int, int> tileShape,
                int zoom, std::pair<float, float> range) -> bmp_t {
                 // make a channel
                 pyre::journal::debug_t channel("ampcor.viz.slc");

                 // unpack the range
                 auto [minv, maxv] = range;

                 // pick the number of bins
                 int bins = 1 << 5;
                 // pick a saturation
                 auto saturation = 0.8;
                 // pick a brightness range
                 auto minB = 0.2;
                 auto maxB = 1.0;

                 // isolate the portion of the raster i care about
                 slc_t::index_type boxOrigin = { tileOrigin.first, tileOrigin.second };
                 slc_t::shape_type boxShape = { tileShape.first, tileShape.second };
                 // make the tile
                 auto tile = slc.box(boxOrigin, boxShape);
                 // make an iterator to its beginning
                 auto start = tile.cbegin();

                 // make the color map
                 slc_complex_t cmap(start,
                                    bins,
                                    saturation,
                                    minB,maxB,
                                    minv, maxv);

                 // make a bitmap object
                 bmp_t bmp(boxShape[0], boxShape[1]);
                 // encode the data using the color map
                 bmp.encode(cmap);

                 // and return it
                 return bmp;
             },
             // the signature
             "raster"_a, "origin"_a, "shape"_a, "zoom"_a, "range"_a,
             // the doctring
             "make a bitmap out the specified SLC tile"
             )

        // generate a bitmap out of the amplitude of an SLC tile
        .def("slcAmplitude",
             // the handler
             [](slc_const_reference slc,
                std::pair<int, int> tileOrigin, std::pair<int, int> tileShape,
                int zoom, std::pair<float, float> range) -> bmp_t {
                 // make a channel
                 pyre::journal::debug_t channel("ampcor.viz.slc");

                 // unpack the range
                 auto [minv, maxv] = range;

                 // pick the number of bins
                 int bins = 1 << 5;
                 // pick hue and saturation
                 auto hue = 0;
                 auto saturation = 0;
                 // pick a brightness range
                 auto minB = 0.2;
                 auto maxB = 1.0;

                 // isolate the portion of the raster i care about
                 slc_t::index_type boxOrigin = { tileOrigin.first, tileOrigin.second };
                 slc_t::shape_type boxShape = { tileShape.first, tileShape.second };
                 // make the tile
                 auto tile = slc.box(boxOrigin, boxShape);
                 // make an iterator to its beginning
                 auto start = tile.cbegin();
                 // and an amplitude calculator
                 ampcor::viz::slc_detector_t detector(start);

                 channel
                     << pyre::journal::newline
                     << "origin: " << boxOrigin << pyre::journal::newline
                     << "shape: " << boxShape << pyre::journal::newline
                     << "bins: " <<  bins << pyre::journal::newline
                     << "hue: " <<  hue << pyre::journal::newline
                     << "saturation: " <<  saturation << pyre::journal::newline
                     << "minimum brightness: " <<  minB << pyre::journal::newline
                     << "maximum brightness: " <<  maxB << pyre::journal::newline
                     << "minimum value: " << minv << pyre::journal::newline
                     << "maximum value: " << maxv << pyre::journal::newline
                     << pyre::journal::endl;
                 // make the color map
                 slc_uni1d_t cmap(detector,
                                  bins,
                                  hue, saturation,
                                  minB,maxB,
                                  minv, maxv);

                 // make a bitmap object
                 bmp_t bmp(boxShape[0], boxShape[1]);
                 // encode the data using the color map
                 bmp.encode(cmap);

                 // and return it
                 return bmp;
             },
             // the signature
             "raster"_a, "origin"_a, "shape"_a, "zoom"_a, "range"_a,
             // the doctring
             "make a bitmap out the specified SLC tile"
             )

        // generate a bitmap out of the amplitude of an SLC tile
        .def("slcPhase",
             // the handler
             [](slc_const_reference slc,
                std::pair<int, int> tileOrigin, std::pair<int, int> tileShape,
                int zoom) -> bmp_t {
                 // make a channel
                 pyre::journal::debug_t channel("ampcor.viz.slc");

                 // pick the number of bins
                 int bins = 1 << 5;
                 // pick saturation and brightness
                 auto saturation = 0.5;
                 auto brightness = 0.5;

                 // isolate the portion of the raster i care about
                 slc_t::index_type boxOrigin = { tileOrigin.first, tileOrigin.second };
                 slc_t::shape_type boxShape = { tileShape.first, tileShape.second };
                 // make the tile
                 auto tile = slc.box(boxOrigin, boxShape);
                 // make an iterator to its beginning
                 auto start = tile.cbegin();
                 // and a phase calculator
                 ampcor::viz::slc_phaser_t phaser(start);

                 // make the color map
                 slc_phase1d_t cmap(phaser, bins, saturation, brightness);

                 // make a bitmap object
                 bmp_t bmp(boxShape[0], boxShape[1]);
                 // encode the data using the color map
                 bmp.encode(cmap);

                 // and return it
                 return bmp;
             },
             // the signature
             "raster"_a, "origin"_a, "shape"_a, "zoom"_a,
             // the doctring
             "make a bitmap out the specified SLC tile"
             )

        // done
        ;


    // all done
    return;
}


// end of file
