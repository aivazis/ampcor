// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_correlators_Sequential_h)
#define ampcor_correlators_Sequential_h


// a worker that executes a correlation plan one tile at a time
template <class productT>
class ampcor::correlators::Sequential {
    // types
public:
    // my template parameter
    using product_type = productT;
    using product_const_reference = const product_type &;
    // the product spec
    using spec_type = typename product_type::product_type;
    // tile shape
    using shape_type = typename spec_type::shape_type;
    using shape_const_reference = const shape_type &;

    // the size of things
    using size_type = size_t;

    // metamethods
public:
    Sequential(size_type pairs,
               shape_const_reference ref, shape_const_reference sec,
               size_type refineFactor, size_type refineMargin,
               size_type zoomFactor);

    // default metamethods
public:
    // destructor
    ~Sequential() = default;
    // constructors
    Sequential(const Sequential &) = default;
    Sequential(Sequential &&) = default;
    Sequential & operator=(const Sequential &) = default;
    Sequential & operator=(Sequential &&) = default;
};


# endif

// end of file
