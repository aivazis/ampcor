// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2022 all rights reserved


// get colors
import { wheel, theme } from '~/palette'
// get the base styles
import base from '~/views/styles'


// publish
export default {
    // the container
    exp: {
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

    canvas: {
        // this must be sized by the code based on the raster shape
        overflow: "hidden",   // hide anything that sticks out
        width: "1000px",
        height: "1000px",
        // for my children
        display: "flex",
        flexDirection: "row",
        flexWrap: "wrap",
    },

    tile: {
        // visibility: "hidden",
        flex: "none",
        width: "500px",
        height: "500px",
    },

    plot: {
        flex: "none",
        width: "500px",
        height: "500px",
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
