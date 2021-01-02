// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2021 all rights reserved


// externals
import React from 'react'
// locals
import styles from './styles'
import { useFlowContext } from '~/context'


// the ampcor workflow
const flow = () => {
 
    // build the container and return it
    return (
        <section style={styles.flow}>
            <div style={styles.placeholder}>
                the flow
            </div>
        </section>
    )
}


// publish
export default flow


// end of file
