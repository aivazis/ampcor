// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2020 all rights reserved


// externals
#include "external.h"
// namespace setup
#include "forward.h"


// grid + memory
#include <p2/grid.h>
// libampcor
#include <ampcor/dom.h>
#include <ampcor/correlators.h>

// type aliases
using slc_t = ampcor::dom::slc_const_t;
using sequential_t = ampcor::correlators::sequential_t<slc_t>;


// add bindings to the sequential correlator
void
ampcor::py::
sequential(py::module &m) {
    // the SLC interface
    py::class_<sequential_t>(m, "Sequential")
        // done
        ;

    // all done
    return;
}


// end of file
