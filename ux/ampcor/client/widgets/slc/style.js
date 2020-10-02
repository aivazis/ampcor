// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// get colors
import { wheel, theme } from '~/palette'


// publish
export default {
    tool: {
        cursor: "pointer",
        width: "4.0em",
        height: "4.0em",
    },

    frame: {
        stroke: wheel.gray.aluminum,
        strokeWidth: "2px",
    },

    shape: {
        stroke: theme.banner.name,
        strokeWidth: "5px",
    },

    selected: {
        stroke: theme.banner.name,
    },

}


// end of file
