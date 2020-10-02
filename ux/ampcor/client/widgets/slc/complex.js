// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import React from 'react'
// locals
import base from './style'
import Frame from './frame'


// the bar at the bottom of every page
const widget = ({selected, selector, style, xform}) => {
    // assemble the tool style
    const toolStyle = { ...base.tool, ...style.tool }
    // assemble the frame style
    const frameStyle = { ...base.frame, ...style.frame }
    // and the shape style
    const shapeStyle = { ...base.shape, ...style.shape }

    // check whether i should highlight my border
    const highlight = selected === "cmplx" ? true : false

    // render
    return (
        // the container
        <svg version="1.1" xmlns="http://www.w3.org/2000/svg" style={{...toolStyle}}>
            {/* box */}
            <g transform={xform} onClick={selector} >
                {/* the frame */}
                <Frame highlight={highlight} style={frameStyle} />
                {/* the tool */}
                <g {...shapeStyle}>
                    <path d="M 37 28 v 45 M 48 24 v 53" />
                    <path d="M 71.5 37.5 A 25 27 0 1 0 71.5 62.5" />
                </g>
            </g>
        </svg>
    )
}


// publish
export default widget


// end of file
