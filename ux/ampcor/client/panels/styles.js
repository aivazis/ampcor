// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2021 all rights reserved


// get colors
import { wheel, theme } from '~/palette'


// publish
export default {

    // the outer container; it owns and positions the toolboxes and must not permit scrolling,
    // otherwise the controls will not be at a fixed location
    panel: {
        // put me wherever, but let me own the positining of my children
        position: "relative",
        // i'm resizable
        flex: 1,

        // make me the same color as the page
        backgroundColor: theme.page.background,

        // manage my children
        overflow: "hidden",    // no scrolling
        display: "flex",       // most likely, i only have one child that cares, but just in case
        flexDirection: "row",
    },

    // for panels that are under construction
    placeholder: {
        position: "fixed",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
    },
}


// end of file
