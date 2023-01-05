// -*- C++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2023 all rights reserved


// externals
#include "external.h"
// namespace setup
#include "forward.h"
// libampcor
#include <ampcor/dom.h>


// bindings for correlation plans
// a correlation plan is a tuple whose elements are pairs of {layout2d} objects that relate
// a tile in the reference raster to a tile in the secondary raster
void
ampcor::py::plan(py::module & m)
{
    // cover {bounds} with a grid of {shape} uniformly spaced points
    m.def(
        // the name
        "uniformGrid",
        // the implementation
        [](const shape2d_t & bounds, const shape2d_t & shape) -> points_t {
            // make a timer
            proctimer_t t("ampcor.plan.uniformGrid");
            // and start it
            t.start();

            // unpack the bounds
            auto [b0, b1] = bounds;
            // and the shape
            auto [s0, s1] = shape;

            // split {bounds} into evenly spaced tiles
            auto t0 = b0 / (s0 + 1);
            auto t1 = b1 / (s1 + 1);
            // compute the unallocated border around the raster
            auto m0 = b0 % (s0 + 1);
            auto m1 = b1 % (s1 + 1);

            // make a pile
            points_t points;
            // go through the 0 shape of the point grid
            for (auto i0 = 0; i0 < s0; ++i0) {
                // build the 0 coordinate of the point
                auto idx0 = m0 / 2 + (i0 + 1) * t0;
                // go through the 1 shape of the grid
                for (auto i1 = 0; i1 < s1; ++i1) {
                    // build the 1 coordinate of the point
                    auto idx1 = m1 / 2 + (i1 + 1) * t1;
                    // make an index
                    index2d_t idx { idx0, idx1 };
                    // add it to the pile
                    points.push_back(idx);
                }
            }

            // stop the timer
            t.stop();

            // make a channel
            auto channel = pyre::journal::debug_t("ampcor.plan.uniformGrid");
            // and show me
            channel << "domain: " << t.ms() << " ms" << pyre::journal::endl(__HERE__);

            // return the sequence of points
            return points;
        },
        // the signature
        "bounds"_a, "shape"_a,
        // the docstring
        "cover {bounds} with a grid of {shape} uniformly spaced points");

    // functor that adds a constant shift to a collection of points
    m.def(
        // the name
        "constantShift",
        // the implementation
        [](const points_t & points, const index2d_t shift) -> plan_t {
            // make a timer
            proctimer_t t("ampcor.plan.constantShift");
            // and start it
            t.start();

            // make a pile
            plan_t pile;
            // go through the points
            for (auto pid = 0; pid < points.size(); ++pid) {
                // get each one
                const auto & p = points[pid];
                // shift it
                auto shifted = p + shift;
                // make a pairing
                pairing_t pair { pid, p, shifted };
                // and add it to the pile
                pile.push_back(pair);
            }

            // stop the timer
            t.stop();

            // make a channel
            auto channel = pyre::journal::debug_t("ampcor.plan.constantShift");
            // and show me
            channel << "range: " << t.ms() << " ms" << pyre::journal::endl(__HERE__);

            // return the pile
            return pile;
        },
        // the signature
        "points"_a, "shift"_a,
        // the docstring
        "apply a constant {shift} to a collection of {points}");
}

// end of file
