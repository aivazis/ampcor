// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// externals
import React from 'react'
import Link from 'react-router-dom'
import Kill from '~/widgets/kill'
// locals
import styles from './styles'


// the top bar
const header = () => (
    // the container
    <header style={styles.header}>
        {/* the application name */}
        <div style={styles.app}>
            ampcor
        </div>

        {/* the kill button */}
        <Kill style={styles.kill} transform="scale(0.1 0.1)"/>

    </header>
)


// publish
export default header


// end of file
