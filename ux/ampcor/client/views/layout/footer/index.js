// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// externals
import React from 'react'
import Flame from '~/widgets/flame'
// locals
import styles from './styles'


// the bar at the bottom of every page
const footer = () => (
    // the container
    <footer style={styles.footer}>
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
