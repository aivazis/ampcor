// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// get colors
import { wheel, theme } from '~/palette'


// publish
export default {
    viewport: {
        // hug my parent
        padding: "0.0em",
        margin: "0.0em",
        width: "100%",
        height: "100%",
        // scroll
        overflow: "auto",
    },

    mosaic: {
        // this must be sized by the code based on the raster shape
        overflow: "hidden",   // hide anything that sticks out
        // let flex position my children
        display: "flex",
        // the current implementation is jsut a list of tiles that getwrapped basedon their size
        flexDirection: "row",
        flexWrap: "wrap",
    },
}


// end of file
