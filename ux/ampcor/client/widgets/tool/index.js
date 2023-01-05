// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2023 all rights reserved


// externals
import React from 'react'
// locals
import styles from './style'


// the tool
const tool = ({state, style, click, children}) => {
    // build the event handler
    const onClick = (state === "active") ? click : null

    // mix the container styles
    const boxStyle = {
        ...styles.base.box, ...styles?.[state].box,
        ...style?.base?.box, ...style?.[state]?.box
    }
    // mix the svg container styles
    const svgStyle = {
        ...styles.base.svg, ...styles?.[state].svg,
        ...style?.base?.svg, ...style?.[state]?.svg
    }
    // mix the top level group styles
    const groupStyle = {
        ...styles.base.group, ...styles?.[state].group,
        ...style?.base?.group, ...style?.[state]?.group
    }
    // mix the frame styles
    const frameStyle = {
        ...styles.base.frame, ...styles?.[state].frame,
        ...style?.base?.frame, ...style?.[state]?.frame
    }
    // mix the icon styles
    const iconStyle = {
        ...styles.base.icon, ...styles?.[state].icon,
        ...style?.base?.icon, ...style?.[state]?.icon
    }

    // render
    return (
        <div style={boxStyle} onClick={onClick} >
            <svg version="1.1" xmlns="http://www.w3.org/2000/svg" {...svgStyle} >
                {/* box */}
                <g {...groupStyle}>
                    {/* the frame */}
                    <rect x="5" y="5" width="90" height="90" {...frameStyle} />
                    {/* the tool */}
                    <g {...iconStyle}>
                        {children}
                    </g>
                </g>
            </svg>
        </div>
    )
}


// publish
export default tool


// end of file
