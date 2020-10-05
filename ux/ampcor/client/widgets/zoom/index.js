// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import React, { useState } from 'react'
// locals
import styles from './style'
import In from './in'
import Out from './out'


// the zoom toolbox
const widget = ({zoomin, zoomout, style}) => {
    // mix the toolbox style
    const toolboxStyle = { ...styles.toolbox, ...style.toolbox }

    // render
    return (
        <div style={toolboxStyle}>
            <In style={style?.tool} click={zoomin} />
            <Out style={style?.tool} click={zoomout} />
        </div>
    )
}


// publish
export default widget


// end of file
