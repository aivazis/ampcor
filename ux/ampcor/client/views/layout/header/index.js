// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// externals
import React, { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import Kill from '~/widgets/kill'
// locals
import styles from './styles'


// the top bar
const header = () => {
    // tie the visibility of header elements to my state
    const [visibility, setVisibility] = useState("visible")
    // make a function that sets the visibility to {hidden}; we will hand this to the {kill} widget
    const hide = () => setVisibility("hidden")

    // get the {nav} style and adjust the visibility of its container
    var navStyle = {...styles.nav.box, visibility}
    // get the {kill} widget styling
    var killStyle = {...styles.kill}
    // and adjust the visibility of its container as well
    killStyle.box.visibility = visibility

    // get the current page
    const location = useLocation().pathname

    // assemble the style for the link
    const decorate = name => (
        location.startsWith('/' + name) ?
        {...styles.nav.name, ...styles.nav.current} : styles.nav.name
    )

    // build the component and return it
    return (
        // the container
        <header style={styles.header}>
            {/* the application name */}
            <div style={styles.app}>
                ampcor
            </div>

            {/* the menu */}
            <nav style={navStyle}>
                <Link to="/flow" style={styles.nav.link}>
                    <span style={decorate("flow")}
                          title="configure the ampcor workflow">
                        flow
                    </span>
                </Link>
                <Link to="/exp" style={styles.nav.link}>
                    <span style={decorate("exp")}
                          title="a sample tiled plot">
                        exp
                    </span>
                </Link>
                <Link to="/slc" style={styles.nav.link}>
                    <span style={decorate("slc")}
                          title="display the input rasters">
                        slc
                    </span>
                </Link>
                <Link to="/offsets" style={styles.nav.link}>
                    <span style={decorate("offsets")}
                          title="visualize the estimated offsets">
                        offsets
                    </span>
                </Link>
                <Link to="/plan" style={styles.nav.link}>
                    <span style={decorate("plan")}
                          title="show the correlation plan" >
                        plan
                    </span>
                </Link>
                <Link to="/gamma" style={{...styles.nav.link, ...styles.nav.last}}>
                    <span style={decorate("gamma")}
                          title="visualize the correlaton surface">
                        gamma
                    </span>
                </Link>
            </nav>

            {/* the kill button */}
            <Kill style={killStyle} onKill={hide} transform="scale(0.1 0.1)"/>

        </header>
    )
}


// publish
export default header


// end of file
