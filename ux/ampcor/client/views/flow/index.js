// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import React from 'react'
import { graphql, useFragment } from 'react-relay/hooks'
// locals
import styles from './styles'
import { useFlowContext } from '~/context'


// the ampcor workflow
const flow = () => {
    // get the flow configuration
    const config = useFlowContext()

    // unpack
    // the flow
    const { name: flowName, family: flowFamily } = config
    // the reference slc
    const { name: refName, family: refFamily, shape: refShape} = config.reference
    // and the secondary slc
    const { name: secName, family: secFamily, shape: secShape} = config.secondary

    // build the container and return it
    return (
        <section style={styles.flow}>
            <div style={styles.placeholder}>
                <ul>
                    <li>flow: {flowName}</li>
                    <li>ref: {refName}</li>
                    <li>sec: {secName}</li>
                </ul>
            </div>
        </section>
    )
}


// publish
export default flow


// end of file
