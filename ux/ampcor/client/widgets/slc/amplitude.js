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
    const name = "amplitude"
    // so choose me when i'm clicked
    const pick = () => click(name)

    // render
    return (
        // the container
        <Tool click={pick} state={state[name]} {...rest}>
            {/* my icon */}
            <path d="M 28 25 v 50 M 72 25 v 50" />
            <circle cx="50" cy="50" r="5" />

        </Tool >
    )
}


// publish
export default widget


// end of file
