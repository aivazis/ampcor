// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2022 all rights reserved


// externals
import React from 'react'
// locals
import styles from './style'
import In from './in'
import Out from './out'


// the zoom toolbox
const widget = ({style}) => {
    // mix the toolbox style
    const toolboxStyle = { ...styles.toolbox, ...style.toolbox }
    // and render
    return (
        <div style={toolboxStyle}>
            <In style={style?.tool} />
            <Out style={style?.tool} />
        </div>
    )
}


// publish
export default widget


// end of file
