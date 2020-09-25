// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// externals
import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import Kill from '~/widgets/kill'
// locals
import styles from './styles'


// assemble the style for the link
const decorate = (name, location) => {
    return location.startsWith('/' + name) ?
           {...styles.nav.name, ...styles.nav.current} : styles.nav.name
}


// the top bar
const header = () => {
    // get the current page
    const location = useLocation().pathname

    // build the component and return it
    return (
        // the container
        <header style={styles.header}>
            {/* the application name */}
            <div style={styles.app}>
                ampcor
            </div>

            {/* the menu */}
            <nav style={styles.nav}>
                <Link to="/flow" style={styles.nav.link}>
                    <span style={decorate("flow", location)}
                          title="configure the ampcor workflow">
                        flow
                    </span>
                </Link>
                <Link to="/exp" style={styles.nav.link}>
                    <span style={decorate("exp", location)}
                          title="a sample tiled plot">
                        exp
                    </span>
                </Link>
                <Link to="/slc" style={styles.nav.link}>
                    <span style={decorate("slc", location)}
                          title="display the input rasters">
                        slc
                    </span>
                </Link>
                <Link to="/offsets" style={styles.nav.link}>
                    <span style={decorate("offsets", location)}
                          title="visualize the estimated offsets">
                        offsets
                    </span>
                </Link>
                <Link to="/plan" style={styles.nav.link}>
                    <span style={decorate("plan", location)}
                          title="show the correlation plan" >
                        plan
                    </span>
                </Link>
                <Link to="/gamma" style={{...styles.nav.link, ...styles.nav.last}}>
                    <span style={decorate("gamma", location)}
                          title="visualize the correlaton surface">
                        gamma
                    </span>
                </Link>
            </nav>

            {/* the kill button */}
            <Kill style={styles.kill} transform="scale(0.1 0.1)"/>

        </header>
    )
}


// publish
export default header


// end of file
