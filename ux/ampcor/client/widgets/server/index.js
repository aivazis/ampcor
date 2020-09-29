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


// display the server state
const server = ({version, style}) => {
    // the query fragment i care about
    const data = useFragment(
        graphql`fragment server_version on Version {
            major
            minor
            micro
            revision
        }`,
        version
    )

    // merge the overall styles
    const base = {...style.box, ...styles.box, ...style.text, ...styles.text}
    // and the state colorization
    const statusGood = {...style.status.good, ...styles.status.good}

    // get the time
    const now = new Date()
    // use it to make a timestamp
    const title = `last checked on ${now.toString()}`

    // unpack
    const { major, minor, micro, revision } = data

    // build the componnent and return it
    return (
        <div style={{...base, ...statusGood}} title={title}>
            ampcor server {major}.{minor}.{micro} rev {revision}
        </div>
    )
}


// publish
export default server


// end of file
