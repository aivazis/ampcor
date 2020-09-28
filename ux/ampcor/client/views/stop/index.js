// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import React from 'react'
// locals
import styles from './styles'


// the area
const stop = (props) => {
    // ask the server to shut down
    fetch('/stop')

    // the container
    return (
        <section style={styles.stop}>
            <div style={styles.placeholder}>
                ampcor has shut down; please close this window
            </div>
        </section>
    )
}


// publish
export default stop


// end of file
