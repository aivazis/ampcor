// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// externals
import React from 'react'
// locals
import base from './style'


// the x
const mark = `M 0 0 L 100 100 M 0 100 L 100 0`


// the bar at the bottom of every page
const widget = ({style, ...xforms}) => (
    // the action
    <a style={{...base.action, ...style.action}} href="/actions/meta/stop"
       title="kill the server; you'll have to close this window yourself, though"
    >
        {/* the box */}
        <svg version="1.1" xmlns="http://www.w3.org/2000/svg" style={{...base.box, ...style.box}}>
            {/* the shape */}
            <g {...base.shape} {...style.shape} {...xforms}>
                <path d={mark}/>
            </g>
        </svg>
    </a>
)


// publish
export default widget


// end of file
