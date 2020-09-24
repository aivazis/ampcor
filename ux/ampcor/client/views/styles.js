// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// get colors
import { wheel, theme } from '~/palette'


// publish
export default {

    panel: {
        // my box
        flex: "1",
        margin: "1.0em",
        padding: "0.0em",

        // styling
        backgroundColor: theme.page.background,

        // my children
        overflow: "hidden",
        display: "flex",
        flexDirection: "row",
    },
}


// end of file
