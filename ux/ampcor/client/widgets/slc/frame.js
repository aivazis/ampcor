// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import React from 'react'
// locals
import base from './style'


// the frame
const frame = ({highlight, style}) => {
    // check whether i need to be styled as selected
    const extra = highlight ? base.selected : {}
    // build the style
    const frameStyle = { ...style, ...extra }

    // render
    return (
        <g {...frameStyle}>
            <rect x="5" y="5" width="90" height="90" rx="10" ry="10" />
        </g>
    )
}


// publish
export default frame


// end of file
