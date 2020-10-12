// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// get colors
import { wheel, theme } from '~/palette'
// get the base styles
import base from '~/views/styles'


// publish
export default {
    // the container
    slc: {
        // inherit
        ...base.panel,
    },

    placeholder: {
        // inherit
        ...base.placeholder,
    },

    sourceToolbox: {
        // positioning
        position: "absolute",
        zIndex: 100,
        left: "1.0em",
        top: "0.75em",
        // size
        width: "4.2em",
        height: "2.0em",
        // styling
        opacity: "0.75",
    },

}


// end of file
