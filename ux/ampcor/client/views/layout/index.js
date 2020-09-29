// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// externals
import React from 'react'
import { BrowserRouter as Router, Switch, Route } from 'react-router-dom'


// locals
// styles
import styles from './styles'
// common to all pages
import Header from './header'
import Footer from './footer'
// landing pages
import Flow from '../flow'
import EXP from '../exp'
import SLC from '../slc'
import Gamma from '../gamma'
import Offsets from '../offsets'
import Plan from '../plan'
import Loading from '../loading'
import Stop from '../stop'


// the layout
const Layout = ({version}) => (
    <Router>
        <div style={styles.layout}>
            <Header/>
            <Switch>
                {/* the top level views */}
                <Route path="/flow" component={Flow}/>
                <Route path="/exp" component={EXP}/>
                <Route path="/slc" component={SLC}/>
                <Route path="/gamma" component={Gamma}/>
                <Route path="/offsets" component={Offsets}/>
                <Route path="/plan" component={Plan}/>
                <Route path="/loading" component={Loading}/>
                {/* the closing page */}
                <Route path="/stop" component={Stop}/>
                {/* default landing spot */}
                <Route path="/" component={Flow}/>
            </Switch>
            <Footer version={version}/>
        </div>
    </Router>
)


// publish
export default Layout;


// end of file
