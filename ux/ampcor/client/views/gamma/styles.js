// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2025 all rights reserved


// get colors
import { wheel, theme } from '~/palette'
// get the base styles
import base from '~/views/styles'


// publish
export default {
    // the container
    gamma: {
        // inherit
        ...base.panel,
    },

    placeholder: {
        // inherit
        ...base.placeholder,
    },

}


// end of file
