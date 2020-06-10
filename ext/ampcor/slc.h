// -*- C++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//

#if !defined(ampcor_extension_slc_h)
#define ampcor_extension_slc_h


// place everything in my private namespace
namespace ampcor {
    namespace extension {
        namespace slc {

            // compute the size of individual pixels of SLC raster images
            extern const char * const pixelSize__name__;
            extern const char * const pixelSize__doc__;
            PyObject * pixelSize(PyObject *self, PyObject *args);

            // memory map an SLC file
            extern const char * const map__name__;
            extern const char * const map__doc__;
            PyObject * map(PyObject *self, PyObject *args, PyObject *kwds);

            // fetch data at offset
            extern const char * const getitem__name__;
            extern const char * const getitem__doc__;
            PyObject * getitem(PyObject *self, PyObject *args);

        } // of namespace slc`
    } // of namespace extension`
} // of namespace ampcor

#endif

// end of file
