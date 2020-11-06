// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//

// configuration
#include <portinfo>
// ampcor
#include <ampcor_cuda/correlators.h>


// type aliases
// device memory
using heap_t = ampcor::cuda::correlators::cudaheap_t<float>;


// an allocation function
auto alloc(size_t cells)
{
    // make a shared pointer to CUDA device memory block and return it by value
    return heap_t { cells };
}


// use a device allocation
auto read(heap_t block)
{
    // read from the start
    return block[0];
}


// make a scope for the allocation
void test()
{
    // make a channel
    pyre::journal::info_t channel("ampcor.cuda");

    // sign in
    channel
        << "test: sanity check for cuda device memory allocation"
        << pyre::journal::endl(__HERE__);

    // allocate a block
    auto block = alloc(1024*1024);
    // and use it
    auto value = read(block);

    // show me
    channel
        << "read: " << value
        << pyre::journal::endl(__HERE__);

    // let it go out of scope
    return;
}


// driver
int main() {
    // exercise...
    test();
    // all done
    return 0;
}


// end of file
