// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2021 all rights reserved


// get colors
import { wheel, theme } from '~/palette'
// get the base styles
import base from '~/views/styles'


// publish
export default {
    // the container
    flow: {
        // inherit
        ...base.panel,
    },

    placeholder: {
        // inherit
        ...base.placeholder,
    },

}


// end of file
