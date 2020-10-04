// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


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
            <ellipse cx="50" cy="48" rx="27" ry="17" />
            <path d="M 50 20 v 60" />
        </Tool >
    )
}


// publish
export default widget


// end of file
