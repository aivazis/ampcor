// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// get colors
import { theme } from '~/palette'


// styling for the page container
export default {
    // the top level flex container
    layout : {
        // placement
        width: "100%",
        height: "100%",

        // overall styling
        backgroundColor: theme.page.background,

        // for my children
        display: "flex",
        flexDirection: "column",
    },
}


// end of file
