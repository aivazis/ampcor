// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import React from 'react'
import { graphql, useLazyLoadQuery } from 'react-relay/hooks'

// locals
import styles from './styles'


// display the server state
const server = ({style, ...props}) => {
    // the query fragment i care about
    const { version: {major, minor, micro, revision} } = useLazyLoadQuery(
        graphql`query serverQuery {
            version {
                major
                minor
                micro
                revision
            }
        }`
    )

    // merge the overall styles
    const base = {...style.box, ...styles.box, ...style.text, ...styles.text}
    // and the state colorization
    const good = {...style.status.good, ...styles.status.good}

    // get the time
    const now = new Date()
    // use it to make a timestamp
    const title = `last checked on ${now.toString()}`

    // build the componnent and return it
    return (
        <div style={{...base, ...good}} title={title}>
            ampcor server {major}.{minor}.{micro} rev {revision}
        </div>
    )
}


// publish
export default server


// end of file
