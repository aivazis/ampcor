// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import React, { useState } from 'react'
// locals
import styles from './style'
import Amplitude from './amplitude'
import Complex from './complex'
import Phase from './phase'


// the slc signal toolbox
//   - contains three buttons that toglle between shown amplitude, phase, or the complex signal
//   - assumes the client understand the words "amplitude", "phase", and "complex"
//   - defaults to "amplitude"
//
//   selector: inform the client about the user's choice
const slc = ({select, style}) => {
    // state management
    // the default selection
    const [selection, setSelection] = useState("amplitude")
    // all tools are active by default
    const state = { amplitude: "active", phase: "active", complex: "active" }
    // and one is selected
    state[selection] = "selected"

    // when a tool is clicked, two things have to happen:
    const pick = (selection) => {
        // the local selection must be updated
        setSelection(selection)
        // and we have to notify the client
        select(selection)
    }

    // mix the toolbox style
    const toolboxStyle = { ...styles.toolbox, ...style.toolbox }

    // render
    return (
        <div style={toolboxStyle}>
            <Amplitude state={state} style={style?.tool} click={pick} />
            <Phase state={state} style={style?.tool} click={pick} />
            <Complex state={state} style={style?.tool} click={pick} />
        </div>
    )
}


// publish
export default slc


// end of file
