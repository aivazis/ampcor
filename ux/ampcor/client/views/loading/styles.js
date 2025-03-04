// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2025 all rights reserved


// get colors
import { wheel, theme } from '~/palette'
// get the base styles
import base from '~/views/styles'


// publish
export default {
    // the container
    loading: {
        // inherit
        ...base.panel,
    },

    placeholder: {
        position: "fixed",
        top: "50%",
        left: "50%",
        width: "100%",
        textAlign: "center",
        transform: "translate(-50%, -50%)",
    },

    logo: {
        // placement
        margin: "1.0em auto",
        width: "216px",
        height: "301px",

        // animation
        animationName: "fadeInOut",
        animationDuration: "3s",
        animationIterationCount: "infinite",
    },

    message: {
        fontFamily: "inconsolata",
        fontSize: "120%",
        textAlign: "center",
    },

}


// end of file
