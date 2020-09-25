// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


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

    tile: {
        // visibility: "hidden",
        flex: "none",
        width: "500px",
        height: "500px",
    },

    plot: {
        flex: "none",
        width: "20000px",
        height: "40000px",
        display: "flex",
        flexDirection: "row",
        flexWrap: "wrap",
        overflow: "hidden",
    },

    placeholder: {
        position: "fixed",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
    },

}


// end of file
