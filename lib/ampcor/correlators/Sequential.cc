// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// configuration
#include <portinfo>
// pull the declarations
#include "public.h"


// interface
void
ampcor::correlators::Sequential::
adjust()
{
    // make a timer
    timer_t timer("ampcor.sequential");
    // make a channel
    pyre::journal::info_t channel("ampcor.sequential");
    // and another for reporting timings
    pyre::journal::info_t tlog("ampcor.sequential");

    // compute the size of the search margin
    auto margin = _secShape - _refShape + index_type::fill(1);
    // and figure out how many cells this is
    auto ccells = std::accumulate(margin.begin(), margin.end(), 1, std::multiplies<size_type>());
    // allocate memory for the correlation results
    _correlation = new double [ _pairs * ccells ] ;
    // stop the clock

    // go through all the pairs
    for (auto pid = 0; pid < _pairs; ++pid) {
        // start the clock
        timer.reset().start();

        // place a grid for the reference tile over our arena
        gview_type ref { {_refShape}, _buffer + pid*(_refCells + _secCells) };
        // place a grid for the secondary search window over our arena
        gview_type sec { {_secShape}, _buffer + pid*(_refCells + _secCells) + _refCells };

        // place a grid over the results area
        gview_type cor { {margin}, _correlation + pid*ccells };

        // build a sum area table for the secondary grid
        sat_type sat(sec.layout());

        // make a view over the reference tile
        auto rView = ref.view();
        // so we compute the variance of the reference tile
        auto rVar = std::inner_product(rView.begin(), rView.end(), rView.begin(), 0.0);

        // for each spot in the correlation matrix
        for (auto anchor : cor.layout()) {
            // form a slice of the secondary search window that has the same shape as the
            // reference tile but is anchored at {anchor}
            auto slice = sec.layout().slice(anchor, anchor+_refShape);
            // compute the average amplitude within this slice
            auto tAvg = sat.sum(slice) / _refCells;

            // initialize the numerator
            cell_type num = 0.0;
            // initialize the variance of the secondary tile
            cell_type tVar = 0.0;

            // use the reference tile as the source of index counters
            for (auto idx : ref.layout()) {
                // get the value from the reference tile
                auto rValue = ref[idx];
                // get the value from the secondary tile and subtract the average amplitude
                auto tValue = sec[anchor+idx] - tAvg;
                // update the numerator
                num += rValue * tValue;
                // update the secondary variance
                tVar += tValue * tValue;
            }

            // store the correlation value
            cor[anchor] = num / std::sqrt(rVar * tVar);
        }

        // stop the clock
        timer.stop();
        // show me
        tlog
            << pyre::journal::at(__HERE__)
            << "pair #" << pid << ": correlating: " << 1e6 * timer.read() << " μs"
            << pyre::journal::endl;
    }

    // all done
    return;
}

void
ampcor::correlators::Sequential::
refine()
{
    return;
}


// meta-methods
ampcor::correlators::Sequential::
~Sequential() {
    delete [] _buffer;
    delete [] _correlation;
}

ampcor::correlators::Sequential::
Sequential(size_type pairs, const shape_type & refShape, const shape_type & secShape) :
    _pairs(pairs),
    _refShape(refShape),
    _secShape(secShape),
    _refCells(std::accumulate(refShape.begin(), refShape.end(), 1, std::multiplies<size_type>())),
    _secCells(std::accumulate(secShape.begin(), secShape.end(), 1, std::multiplies<size_type>())),
    _buffer { new double [ _pairs*(_refCells+_secCells) ] },
    _correlation { nullptr }
{
    // compute the footprint
    auto footprint = _pairs*(_refCells + _secCells);

    // make a channel
    pyre::journal::debug_t channel("ampcor.sequential");
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "new sequential worker:" << pyre::journal::newline
        << "    pairs: " << _pairs << pyre::journal::newline
        << "    ref shape: " << _refShape << pyre::journal::newline
        << "    sec shape: " << _secShape << pyre::journal::newline
        << "    footprint: " << footprint << " cells in " << (8.0*footprint/1024/1024) << " Mb"
        << pyre::journal::endl;
}


// end of file
