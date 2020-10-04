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
    // render
    return (
        // the container
        <Tool click={click} state="active" {...rest}>
            {/* my icon */}
            <path d="M 25 50 h 50" />
        </Tool >
    )
}


// publish
export default widget


// end of file
