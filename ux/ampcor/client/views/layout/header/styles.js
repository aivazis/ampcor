// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// get the colors
import { wheel, theme } from '~/palette'


// publish
export default {
    // the overall container
    header: {
        // scale 'em" down
        fontSize: "50%",

        // my box
        flex: "none",
        height: "45px",
        margin: "0.25em 1.0em auto 1.0em",
        padding: "0.25em",

        // styling
        backgroundColor: theme.banner.background,
        borderBottom: `1px solid ${theme.banner.separator}`,

        // for my children
        display: "flex",
        flexDirection: "row",
    },

    // the application title box
    app: {
        // fornts
        fontFamily: "georgia, \"times new roman\", serif",
        fontStyle: "italic",
        fontSize: "200%",

        // placement
        // don't let me grow
        flex: "none",
        margin: "auto 0.5em 0.5em 0.5em",

        // styling
        color: theme.banner.name,
    },

    // the kill button
    kill: {
        // the link
        action: {
            // placement: stick to the right edge
            margin: "auto 0.5em 1.0em 0.5em",
        },
        // the container
        box: {},
        // rendering
        path: {},
    },

    // navigation
    nav: {

        // the container
        box: {
            // placement
            margin: "auto auto 1.0em 0.5em",

            // for my children
            display: "flex",
            flexDirection: "row",
        },

        link: {
            // placement
            padding: "0.0em 1.5em 0.0em 1.5em",
            color: theme.banner.nav.link,
            // borderRight: `1px solid ${theme.banner.nav.separator}`,
        },

        // the link to the current page get decorated slightly differently
        current: {
            color: theme.banner.nav.current,
            borderBottom: `1px dotted ${theme.banner.nav.current}`,
        },

        last: {
            // borderRight: "none",
        },

        name: {
            // fonts
            fontFamily: "inconsolata",
            fontSize: "140%",
            whiteSpace: "nowrap",
            textTransform: "lowercase",
        },

    },
}


// end of file
