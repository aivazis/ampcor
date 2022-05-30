// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2022 all rights reserved


// externals
#include "external.h"
// namespace setup
#include "forward.h"


// opaque classes do not get auto-converted to and from python
void
ampcor::py::opaque(py::module & m)
{
    // the correlation plan
    py::bind_vector<plan_t>(m, "Plan");
    // collections of indices
    py::bind_vector<points_t>(m, "Points");

    // all done
    return;
}


// end of file
