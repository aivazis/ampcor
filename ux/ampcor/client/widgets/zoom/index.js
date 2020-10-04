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
const slc = ({click, style}) => {
    // on click: do nothing special, for now
    const pick = () => click()

    // mix the toolbox style
    const toolboxStyle = { ...styles.toolbox, ...style.toolbox }

    // render
    return (
        <div style={toolboxStyle}>
            <In style={style?.tool} click={pick} />
            <Out style={style?.tool} click={pick} />
        </div>
    )
}


// publish
export default slc


// end of file
