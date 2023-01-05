// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// (c) 1998-2023 all rights reserved


// externals
// framework
import React from 'react'
// routing
import { Outlet } from 'react-router-dom'

// locals
import styles from './styles'
// views
import { Header, Footer } from '~/views'


// the main app working area
// the layout is simple: the activity bar and activity dependent routing
export const Main = () => {
    // lay out the main page
    return (
        <main style={styles.page} >
            <Header />
            <Outlet />
            <Footer />
        </main>
    )
}


// end of file
