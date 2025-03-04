// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2025 all rights reserved
//

// configuration
#include <portinfo>
// ampcor
#include <ampcor_cuda/correlators.h>


// type aliases
// device memory
using heap_t = ampcor::cuda::correlators::cudaheap_t<float, false>;


// an allocation function
auto alloc(size_t cells)
{
    // make a shared pointer to CUDA device memory block and return it by value
    return heap_t { cells };
}


// read from a device allocation
auto read(heap_t block)
{
    // read from the start
    return block[0];
}


// write to a device allocation
auto write(heap_t block, float value)
{
    // write to the start
    block[0] = value;
    // all done
    return;
}


// make a scope for the allocation
void test()
{
    // make a channel
    pyre::journal::debug_t channel("ampcor.cuda");

    // sign in
    channel
        << "test: sanity check for cuda device memory allocation"
        << pyre::journal::endl(__HERE__);

    // make a value
    float expect = 3.14;
    // allocate a block
    auto block = alloc(1024*1024);
    // write a value
    write(block, expect);
    // and read it
    auto got = read(block);

    // show me
    channel
        << "wrote: " << expect
        << "read: " << got
        << pyre::journal::endl(__HERE__);

    // check
    if (expect != got) {
        // make a channel
        pyre::journal::error_t error("ampcor.cuda.cudaheap");
        // complain
        error
            << "mismatch: wrote: " << expect << ", read: " << got
            << pyre::journal::endl(__HERE__);
    }

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
