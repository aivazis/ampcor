// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2023 all rights reserved


// get colors
import { wheel, theme } from '~/palette'


// publish
export default {
    // the overall container
    footer: {
        // scale 'em" down
        fontSize: "50%",

        // my box
        flex: "none",
        margin: "auto 1.0em 0.25em 1.0em",
        padding: "0.25em",

        // styling
        backgroundColor: theme.colophon.background,
        // borderTop: `1px solid ${theme.colophon.separator}`,

        // my children
        display: "flex",
        flexDirection: "row",
    },

    // the server info
    server: {
        box: {
            // placement
            margin: "auto auto 0.0em 1.0em",
            padding: "0.0em",
        },

        text: {
            // font
            fontFamily: "inconsolata",
            // styling
            color: theme.page.appversion,
            textAlign: "left",
        },

        status: {
            // when everything is ok
            good: {
                color: wheel.pyre.green,
                opacity: "0.5",
            },
            // when there is an error retrieving the state of the server
            error: {
                color: theme.journal.error,
            },
        },
    },

    // the box with copyright note
    colophon: {
        margin: "auto 2.0em 0.0em auto",
        padding: "0.0 0.0em",
        textAlign: "right",
    },

    copyright: {
        fontFamily: "\"helvetica\", \"arial\", \"sans-serif\"",
        color: theme.colophon.copyright,
        fontWeight: "normal",
        fontSize: "100%",
    },

    author: {
        color: theme.colophon.author,
        textTransform: "uppercase",
    },

    // the pyre logo
    logo: {
        // the container
        box: {
            // box
            flex: "none",
            width: "35px",
            height: "45px",
            // placement
            margin: "0.0em 0.5em 0.0em 0.0em",
        },

        // the actual shape
        shape: {
            fillOpacity: "0.5",
        },
    },

    // error messages
    error: {
        // styling
        color: theme.journal.error,
    },
}


// end of file
