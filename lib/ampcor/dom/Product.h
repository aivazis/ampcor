// -*- c++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved

// code guard
#if !defined(ampcor_dom_Product_h)
#define ampcor_dom_Product_h


// this class takes a product layout and builds a memory mapped grid
template <class specT, bool isReadOnly>
class ampcor::dom::Product : public grid_t<specT, mmap_t, isReadOnly> {
    // type aliases
public:
    // my parameters
    using product_type = specT;
    using product_const_reference = const specT &;
    // my base class
    using grid_type = grid_t<product_type, mmap_t, isReadOnly>;
    // sizes of things
    using size_type = size_t;

    // metamethods
public:
    // constructor
    template <typename... Args>
    constexpr Product(Args&&...);

    // interface
public:
    // {size} is too overloaded, so we use {capacity} to denote the number of cells in the
    // product layout, and {footprint} for its memory requirements
    constexpr auto capacity() const -> size_type;
    constexpr auto footprint() const -> size_type;


    // static interface
public:
    // my read/write flag
    static constexpr auto readOnly() -> bool;
    // my pixel size
    static constexpr auto pixelFootprint() -> size_type;

    // default metamethods
public:
    // destructor
    ~Product() = default;
    // constructors
    Product(const Product &) = default;
    Product(Product &&) = default;
    Product & operator=(const Product &) = default;
    Product & operator=(Product &&) = default;
};


// get the inline definitions
#define ampcor_dom_Product_icc
#include "Product.icc"
#undef ampcor_dom_Product_icc


#endif

// end of file
