// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import React from 'react'
import { graphql, QueryRenderer } from 'react-relay'
import { environment } from '~/context'


// locals
import styles from './styles'


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
    const good = {...style.status.good, ...styles.status.good}
    const error = {...style.status.error, ...styles.status.error}
    const unknown = {...style.status.unknown, ...styles.status.unknown}

    // build the componnent and return it
    return (
        // the server version; retrieved from the server
        <QueryRenderer
            query={q}
            variables={{}}
            environment={environment}
            render={({error, props, ...rest}) => {
                    // get the time
                    const now = new Date()
                    // use it to make a timestamp
                    const title = `last checked on ${now.toString()}`
                    // if something went wrong
                    if (error) {
                        // say so
                        return (
                            <span style={{...base, ...error}}>
                                could not get server version information
                            </span>
                        )
                    }
                    // if no information was passed in
                    if (!props) {
                        // the query hasn't completed yet
                        return (
                            <div style={{...base, ...unknown}}>
                                retrieving version information...
                            </div>
                        )
                    }
                    // otherwise, unpack the version
                    const {major, minor, micro, revision} = props.version
                    // and render it
                    return (
                        <div style={{...base, ...good}} title={title} >
                            ampcor server {major}.{minor}.{micro} rev {revision}
                        </div>
                    )
                }}
        />
    )
}


// publish
export default server


// end of file
