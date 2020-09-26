// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import React from 'react'


// locals
import styles from './styles'
import Query from '~/widgets/query'

// the query that gets me the server state
const q = graphql`
    query serverVersionQuery {
        version {
            major
            minor
            micro
            revision
        }
    }
`

// display the server state
const server = ({style}) => {
    // merge the overall styles
    const base = {...style.box, ...styles.box, ...style.text, ...styles.text}
    // and the state colorization
    const statusGood = {...style.status.good, ...styles.status.good}
    const statusError = {...style.status.error, ...styles.status.error}
    const statusUnknown = {...style.status.unknown, ...styles.status.unknown}

    // get the time
    const now = new Date()
    // use it to make a timestamp
    const title = `last checked on ${now.toString()}`

    // the handlers
    const whileLoading = () => <span style={{...base, ...statusUnknown}}>loading...</span>

    const onError = () => (
        <span style={{...base, ...statusError}}>
            couldn't get server version information
        </span>
    )

    // build the componnent and return it
    return (
        // the server version; retrieved from the server
        <Query query={q} variables={{}} onError={onError} whileLoading={whileLoading}>
            {({version}) => {
                 // unpack the version
                 const {major, minor, micro, revision} = version
                 // render
                 return (
                     <div style={{...base, ...statusGood}} title={title} >
                         ampcor server {version.major}.{version.minor}.{version.micro}
                         rev {version.revision}
                     </div>
                 )
             }}
        </Query>
    )
}


// publish
export default server


// end of file
