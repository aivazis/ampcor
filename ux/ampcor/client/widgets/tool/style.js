// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// get colors
import { wheel, theme } from '~/palette'


// publish
export default {
    base: {
        // style for the container element, most likely a <button>
        box: {
            // override the browser defaults
            cursor: "pointer",
            border: "none",
            fontSize: "20px",
            backgroundColor: theme.banner.separator,
        },

        // the drawing context
        svg: {
            width: "100%",
            height: "100%",
        },

        // transformations that apply to the entire icon; used to scale the drawing down
        // to the size of the client area; thse get spread into the {tranform} attribute
        // of a <g> element
        group: {
        },

        // the border
        frame: {
            fill: "none",
            stroke: "none",
            // smooth the corners
            rx: 5,
            ry: 5,
        },

        // the drawing; these are treated s hints; the icon design overrides
        icon: {
            fill: "none",
            stroke: "none",
        },
    },

    inactive: {
        box: {
        },

        frame: {
        },

        icon: {
        },
    },

    active: {
        box: {
        },

        frame: {
        },

        icon: {
        },
    },

    selected: {
        box: {
        },

        frame: {
        },

        icon: {
        },
    }
}


// end of file
