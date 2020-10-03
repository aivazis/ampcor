// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// get colors
import { wheel, theme } from '~/palette'


// publish
export default {

    // the container
    panel: {
        // my box
        flex: "1",
        position: "relative",
        margin: "1.0em",
        padding: "0.0em",

        // styling
        backgroundColor: theme.page.background,

        // my children
        overflow: "hidden",
        display: "flex",
        flexDirection: "row",
    },

    viewport: {
        padding: "0.0em",
        margin: "0.0em",
        width: "100%",
        height: "100%",
        overflow: "auto",
    },

    // the area with the temporary message for the pages that are under construction
    placeholder: {
        position: "fixed",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
    },
}


// end of file
