// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// externals
import React from 'react'
import Link from 'react-router-dom'
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
        <a style={styles.kill.action} href="/actions/meta/stop">
            <svg style={styles.kill.box} version="1.1" xmlns="http://www.w3.org/2000/svg">
                <g style={styles.kill.path} transform="scale(0.1 0.1)">
                    <path d="M 0 0 L 100 100 M 0 100 L 100 0" />
                </g>
            </svg>
        </a>

    </header>
)


// publish
export default header


// end of file
