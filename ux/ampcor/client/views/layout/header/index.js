// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// externals
import React from 'react'
import { Link } from 'react-router-dom'
import { connect } from 'react-redux'
import Kill from '~/widgets/kill'
// locals
import styles from './styles'


// the top bar
const header = ({page, title}) => {
    // link decorator
    const decorate = (name) => {
        return name === page ? {...styles.nav.name, ...styles.nav.current} : styles.nav.name
    }

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
                    <span style={decorate("flow")}>
                        flow
                    </span>
                </Link>
                <Link to="/slc" style={styles.nav.link}>
                    <span style={decorate("slc")}>
                        slc
                    </span>
                </Link>
                <Link to="/offsets" style={styles.nav.link}>
                    <span style={decorate('offsets')}>
                        offsets
                    </span>
                </Link>
                <Link to="/plan" style={styles.nav.link}>
                    <span style={decorate('plan')} title="show the correlation plan" >
                        plan
                    </span>
                </Link>
                <Link to="/gamma" style={{...styles.nav.link, ...styles.nav.last}}>
                    <span style={decorate('gamma')}>
                        gamma
                    </span>
                </Link>
            </nav>

            {/* the kill button */}
            <Kill style={styles.kill} transform="scale(0.1 0.1)"/>

        </header>
    )
}


// grab the page title from the store
const getPage = ({navigation}) => ({
    page: navigation.get('page'),
    title: navigation.get('title'),
})

// publish
export default connect(getPage)(header)


// end of file
