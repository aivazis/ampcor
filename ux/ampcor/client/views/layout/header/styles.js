// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// get the colors
import { theme } from '~/palette'


// publish
export default {
    // the overall container
    header: {
        // scale 'em" down
        fontSize: "50%",

        // my box
        flex: "none",
        height: "45px",
        margin: "1.0em 1.0em auto 1.0em",
        padding: "0.5em",

        // styling
        backgroundColor: theme.banner.background,
        borderBottom: `1px solid ${theme.banner.separator}`,

        // for my children
        display: "flex",
        flexDirection: "row",
    },

    // the application title box
    app: {
        // placement
        margin: "auto auto auto 0.5em",

        // styling
        fontFamily: "\"georgia\", \"times new roman\", \"serif\"",
        fontStyle: "italic",
        fontSize: "200%",
        color: theme.banner.name,
    },

    // the kill button
    kill: {
        // the link
        action: {
            // placement
            margin: "auto 0.5em auto auto",
        },

        // the ontainer
        box: {},

        // rendering
        path: {},
    },

}


// end of file