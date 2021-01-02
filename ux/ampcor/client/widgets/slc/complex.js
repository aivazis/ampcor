// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2021 all rights reserved


// externals
import React from 'react'
// locals
import Tool from '~/widgets/tool'


// the bar at the bottom of every page
const widget = ({click, state, ...rest}) => {
    // my name is the selection i correspond to
    const name = "complex"
    // so choose me when i'm clicked
    const pick = () => click(name)

    // render
    return (
        // the container
        <Tool click={pick} state={state[name]} {...rest}>
            {/* my icon */}
            <path d="M 71.5 37.5 A 25 27 0 1 0 71.5 62.5" />
            <path d="M 41 26 v 49" />
            {/* <path d="M 37 28 v 45 M 50 24 v 53" /> */}
        </Tool >
    )
}


// publish
export default widget


// end of file
