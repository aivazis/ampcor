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

    viewport: {
        padding: "0.0em",
        margin: "0.0em",
        width: "100%",
        height: "100%",
        border: `1px solid ${wheel.gray.bassalt}`,
        overflow: "auto",
    },

    plot: {
        flex: "none",
        display: "flex",
        flexDirection: "row",
        flexWrap: "wrap",
        overflow: "hidden",
    },

    tile: {
        flex: "none",
        // the sizes here are defaults; they get overriden when more/better is known
        width: "256px",
        height: "256px",
    },

    tools: {
        box: {
            position: "absolute",
            right: "0.5em",
            top: "0.5em",
            width: "2.0em",
            height: "6.0em",
            opacity: "0.75",

            backgroundColor: wheel.gray.white,
            border: `3px solid ${wheel.gray.white}`,

            display: "flex",
            flexDirection: "column",
        },

        tool: {
            flex: "none",
            width: "2.0em",
            height: "2.0em",
        },

        frame: {
        },

        selected: {
        },

        shape: {
        },
    },

    placeholder: {
        // inherit
        ...base.placeholder,
    },

}


// end of file
