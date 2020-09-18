// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// externals
import React from 'react'
import { BrowserRouter as Router } from 'react-router-dom'


// locals
import styles from './styles'
import Header from './header'
import Footer from './footer'


// the layout
const Layout = () => (
    <div style={styles.layout}>
        <Header/>
        <Footer/>
    </div>
)


// publish
export default Layout;


// end of file
