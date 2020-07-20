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
    using spec_type = specT;
    using spec_const_reference = const specT &;
    // me
    using product_type = Product<spec_type, isReadOnly>;
    using product_const_reference = const product_type &;
    // my base class
    using grid_type = grid_t<spec_type, mmap_t, isReadOnly>;
    // my parts
    using storage_pointer = typename grid_type::storage_pointer;
    // my shape
    using shape_type = typename spec_type::shape_type;
    using shape_const_reference = typename spec_type::shape_const_reference;
    // my layout
    using layout_type = typename spec_type::layout_type;
    using layout_const_reference = typename spec_type::layout_const_reference;
    // my index
    using index_type = typename spec_type::index_type;
    using index_const_reference = typename spec_type::index_const_reference;

    // sizes of things
    using size_type = typename spec_type::size_type;

    // metamethods
public:
    // constructor that passes its extra arguments to the storage strategy
    template <typename... Args>
    constexpr Product(spec_const_reference, Args&&...);

    // interface
public:
    // {size} is too overloaded, so we use {cells} to denote the number of cells in the
    // product layout, and {bytes} for its memory requirements
    constexpr auto cells() const -> size_type;
    constexpr auto bytes() const -> size_type;

    // tile factory
public:
    constexpr auto tile(index_const_reference, shape_const_reference) const -> product_type;

    // static interface
public:
    // my read/write flag
    static constexpr auto readOnly() -> bool;
    // my pixel size
    static constexpr auto bytesPerCell() -> size_type;

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
