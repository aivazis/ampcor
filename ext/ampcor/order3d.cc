// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2021 all rights reserved


// externals
#include "external.h"
// namespace setup
#include "forward.h"


// add bindings for the grid layouts used in this package
void
ampcor::py::order3d(py::module & m)
{
    // the order
    auto orderCls = py::class_<order3d_t>(m, "Order3D");

    // populate {Order3D}
    // constructor
    orderCls.def(
        // convert python tuples into indices
        py::init([](std::tuple<int, int> pyOrder) {
            // unpack
            auto [s0, s1] = pyOrder;
            // build an order and return it
            return new layout3d_t::order_type(s0, s1);
        }),
        // the signature
        "order"_a);

    // rank
    orderCls.def_property_readonly_static(
        // the name of the property
        "rank",
        // the getter
        [](py::object) { return order3d_t::rank(); },
        // the docstring
        "the number of entries this order");

    // access to individual ranks
    orderCls.def(
        // the name of the method
        "__getitem__",
        // the getter
        [](const order3d_t & order, int idx) { return order[idx]; },
        // the signature
        "order"_a,
        // the docstring
        "get the value of a given rank");

    // iteration support
    orderCls.def(
        // the name of the method
        "__iter__",
        // the implementation
        [](const order3d_t & order) { return py::make_iterator(order.begin(), order.end()); },
        // the docstring
        "iterate over the ranks",
        // make sure the order lives long enough
        py::keep_alive<0, 1>());

    // string representation
    orderCls.def(
        // the name of the method
        "__str__",
        // the implementation
        [](const order3d_t & order) {
            // make a buffer
            std::stringstream buffer;
            // inject my value
            buffer << order;
            // and return the value
            return buffer.str();
        },
        // the docstring
        "generate a string representation");

    // all done
    return;
}


// end of file
