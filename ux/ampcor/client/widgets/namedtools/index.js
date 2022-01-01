// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2022 all rights reserved


// externals
import React, { useState } from 'react'
// locals
import styles from './style'
import { Tool } from '~/widgets'


// a selector of a prdouct from a "stack"
const stack = ({select, style, choices}) => {
    // state management
    // the default selection
    const [selection, setSelection] = useState(choices[0])
    // all entries are active by default
    const stateTable = choices.reduce(
        (partial, choice) => {
            // mark the current choice as active
            partial[choice] = "active"
            // and return the updated object
            return partial
        },
        // starting with the empty object
        {}
    )
    // and one is selected
    stateTable[selection] = "selected"

    // when a tool is clicked, two things have to happen:
    const pick = (selection) => {
        // the local selection must be updated
        setSelection(selection)
        // and we have to notify the client
        select(selection)
    }

    // mix the toolbox style
    const toolboxStyle = { ...styles.toolbox, ...style }

    // render
    return (
        <div style={toolboxStyle}>
            {choices.map(choice => {
                 // look up the state
                 const state = stateTable[choice]
                 // make an {onClick} callback
                 const choose = () => pick(choice)
                 // render
                 return (
                     <Tool key={choice} click={choose} state={state} style={styles.tool} >
                         <text x="12" y="65">{choice}</text>
                     </Tool>
            )})}
        </div>
    )
}


// publish
export default stack


// end of file
