// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2022 all rights reserved


// externals
import React from 'react'
// locals
import Tool from '~/widgets/tool'


// the bar at the bottom of every page
const widget = ({click, state, ...rest}) => {
    // my name is the selection i correspond to
    const name = "phase"
    // so choose me when i'm clicked
    const pick = () => click(name)

    // render
    return (
        // the container
        <Tool click={pick} state={state[name]} {...rest}>
            {/* my icon */}
            <path d="M 25 50 L 75 25 M 25 50 L 75 75" />
            <path d="M 50 75 A 1 2 0 1 0 50 25" />
        </Tool >
    )
}


// publish
export default widget


// end of file
