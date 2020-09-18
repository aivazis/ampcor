// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// get colors
import { theme } from '~/palette'


// publish
export default {
    // the overall container
    footer : {
        // scale 'em" down
        fontSize: "50%",

        // my box
        flex: "none",
        margin: "auto 1.0em 1.0em 1.0em",
        padding: "0.5em",

        // styling
        backgroundColor: theme.colophon.background,
        borderTop: `1px solid ${theme.colophon.separator}`,

        // my children
        display: "flex",
        flexDirection: "row",
    },

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

    logo: {
        // my box
        flex: "none",
        width: "35px",
        height: "45px",

        margin: "0.0em 0.5em 0.0em 0.0em",
    },
}


// end of file
