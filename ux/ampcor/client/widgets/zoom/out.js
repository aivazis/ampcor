// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2021 all rights reserved


// externals
import React from 'react'
// locals
import { MosaicContext } from '~/context'
import Tool from '~/widgets/tool'


// the bar at the bottom of every page
const widget = ({click, state, ...rest}) => {
    // grab my {zoom} level mutator from the mosaic context provider by my parent
    const { zoomOut } = MosaicContext.useZoom()
    // render
    return (
        // the container
        <Tool click={zoomOut} state="active" {...rest}>
            {/* my icon */}
            <path d="M 25 50 h 50" />
        </Tool >
    )
}


// publish
export default widget


// end of file
