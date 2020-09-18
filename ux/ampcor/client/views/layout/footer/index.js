// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// externals
import React from 'react'
import { graphql, QueryRenderer } from 'react-relay'
import { environment } from '~/context'
import Flame from '~/widgets/flame'
// locals
import styles from './styles'


// the query
const q = graphql`
    query footerVersionQuery {
        version {
            major
            minor
            micro
            revision
        }
    }
`


// the bar at the bottom of every page
const footer = () => (
    // the container
    <footer style={styles.footer}>
        {/* the server version; retrieved from the server */}
        <QueryRenderer
            query={q}
            variables={{}}
            environment={environment}
            render={({error, props, ...rest}) => {
                    // if something went wrong
                    if (error) {
                        // say so
                        return (
                            <div style={{...styles.version, ...styles.error}}>
                               could not get version information
                            </div>
                        )
                    }
                    // if no information was passed in
                    if (!props) {
                        // the query hasn't completed yet
                        return (
                            <div style={styles.version}>
                                       retrieving version information...
                            </div>
                        )
                    }
                    // othetwise, unpack the version
                    const {major, minor, micro, revision} = props.version
                    // and render it
                    return (
                        <div style={styles.version}>
                                   {major}.{minor}.{micor} rev {revision}
                        </div>
                    )
                }}
        />

        {/* the box with the copyright note */}
        <div style={styles.colophon}>
            <span style={styles.copyright}>
                copyright &copy; 1998-2020
                &nbsp;
                <a style={styles.author} href="https://github.com/aivazis">
                    michael&nbsp;aïvázis
                </a>
                &nbsp;
                -- all rights reserved
            </span>
        </div>

        {/* the pyre flame */}
        <Flame style={styles.logo} transform="scale(0.15 0.15)"/>

    </footer>
)


// publish
export default footer


// end of file
