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
    // the container
    plan: {
        // my box
        flex: "1",
        margin: "0.0em 1.0em",
        padding: "0.0em",

        // styling
        backgroundColor: theme.page.background,
        // border: `1px solid ${wheel.granite}`,

        // my children
        display: "flex",
        flexDirection: "row",
    },

    placeholder: {
        position: "fixed",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
    },

}


// end of file
